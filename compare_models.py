"""Qualitative comparison of four mass models for the HFF cluster lenses.

The script deliberately does *not* plot pixel-by-pixel percentage differences.
Such a percentage is not a robust comparison for these products because

* the maps first have to be placed on the same sky grid and angular resolution;
* lensing potential is only defined up to an additive constant and a linear
  term, so its zero point (and therefore a percentage) is arbitrary;
* caustics are curves, not scalar images with a natural percentage difference.

Every potential is reprojected with its FITS WCS onto the common sky area.
Potential and deflection maps are compared at the coarsest native angular
resolution among the selected models, with Gaussian anti-aliasing before a
finer map is downsampled.  Caustics use each model's full footprint but the same
angular resolution, source redshift, and source-plane gauge.
Critical curves and their source-plane ray tracing are computed with
lenstronomy's ``INTERPOL`` model and ``LensModelExtensions``.
Potential units are cross-checked against ``grad(psi) = alpha``;
an affine plane is then removed from each potential *for display only* to remove
the unobservable potential gauge.  The physical deflection fields used for the
caustics are not detrended.  For the caustics, each model's
constant-deflection freedom is fixed at the *same image-plane target position*.
This gives every model the same explicitly defined source-plane origin without
independently translating the caustic curves.

Examples
--------
Compare all four configured models for all clusters::

    python compare_models.py

Compare only Abell 2744 and save PDF as well as PNG::

    python compare_models.py --clusters abell2744 --formats png pdf

Each cluster below has one hard-coded representative simulated-source redshift
and corresponding D_ls/D_s factor.  The identical factor is applied to every
model of that cluster.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from scipy.ndimage import gaussian_filter, map_coordinates


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ClusterSpec:
    """File information and the shared simulated-source lensing geometry."""

    label: str
    directory: str
    file_slug: str
    lens_redshift: float
    source_redshift: float
    deflection_scale: float


@dataclass(frozen=True)
class ModelSpec:
    """How to locate one modelling team's FITS products."""

    label: str
    directory: str
    file_model: str
    version: str


# Representative z_s values are the per-cluster medians in src_pos_tidy_xyz.csv.
# The scales are D_ls/D_s for FlatLambdaCDM(H0=70, Om0=0.3).  The distributed
# HFF deflection/potential products are normalized to D_ls/D_s = 1.
CLUSTERS = {
    "abell370": ClusterSpec(
        "Abell 370", "Abell 370", "abell370", 0.375, 3.196268, 0.776041956
    ),
    "abell2744": ClusterSpec(
        "Abell 2744", "Abell 2744", "abell2744", 0.308, 2.487785, 0.789496503
    ),
    "abells1063": ClusterSpec(
        "Abell S1063", "Abell S1063", "abells1063", 0.351, 3.081273, 0.785609170
    ),
    "macs0416": ClusterSpec(
        "MACS J0416.1-2403", "MACS J0416.1-2403", "macs0416",
        0.397, 2.979514, 0.756764488
    ),
    "macs0717": ClusterSpec(
        "MACS J0717.5+3745", "MACS J0717.5+3745", "macs0717",
        0.545, 2.859944, 0.672878328
    ),
    "macs1149": ClusterSpec(
        "MACS J1149.5+2223", "MACS J1149.5+2223", "macs1149",
        0.543, 3.325716, 0.695150415
    ),
}


# Add future model families here.  ``file_model`` is the token between the
# cluster slug and version in names such as
# hlsp_frontier_model_abell2744_williams_v4_psi.fits.
MODEL_SPECS = {
    "cats": ModelSpec("CATS", "cats copy", "cats", "v4"),
    "williams": ModelSpec("Williams", "william", "williams", "v4"),
    "diego": ModelSpec("Diego", "diego", "diego", "v4"),
    "keeton": ModelSpec("Keeton", "keeton", "keeton", "v4"),
}

MODEL_ALIASES = {
    "cat": "cats",
    "cats-copy": "cats",
    "william": "williams",
}


@dataclass
class RawModel:
    """Native FITS arrays and their shared celestial WCS."""

    spec: ModelSpec
    psi: np.ndarray
    alpha_x: np.ndarray
    alpha_y: np.ndarray
    wcs: WCS
    header: fits.Header
    pixel_scales: tuple[float, float]
    orientation: np.ndarray
    alignment_note: str

    @property
    def pixel_scale(self) -> float:
        """Coarser of the native x/y angular pixel scales."""

        return max(self.pixel_scales)


@dataclass
class PreparedModel:
    """A model sampled on the common west/north angular grid."""

    spec: ModelSpec
    psi: np.ndarray
    alpha_x: np.ndarray
    alpha_y: np.ndarray
    potential_unit_scale: float = 1.0
    potential_fit_error: float = np.nan
    source_gauge_alpha: np.ndarray | None = None
    source_reference_theta: np.ndarray | None = None
    caustics: list[np.ndarray] | None = None
    critical_curves: list[np.ndarray] | None = None


def product_path(data_dir: Path, cluster: ClusterSpec, model: ModelSpec, kind: str) -> Path:
    """Return the expected path of a Frontier Fields lens product."""

    filename = (
        f"hlsp_frontier_model_{cluster.file_slug}_{model.file_model}_"
        f"{model.version}_{kind}.fits"
    )
    return data_dir / cluster.directory / model.directory / filename


def read_fits_array(path: Path) -> tuple[np.ndarray, fits.Header]:
    """Read the primary 2-D image and detach it from the FITS file."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing model product: {path}")
    with fits.open(path, memmap=True) as hdul:
        data = np.array(hdul[0].data, dtype=np.float32, copy=True)
        header = hdul[0].header.copy()
    if data.ndim != 2:
        raise ValueError(f"Expected a 2-D image in {path}, got shape {data.shape}")
    return data, header


def deflection_orientation(wcs: WCS) -> np.ndarray:
    """Map native (x, y) vector components to west/north components.

    Frontier Fields x/y deflection maps use the corresponding native image
    axes.  The columns below are the unit sky directions of increasing native
    x and y pixels.  West is positive here so that the common grid has the same
    handedness as the usual north-up/east-left lens maps.
    """

    matrix = np.asarray(wcs.celestial.pixel_scale_matrix, dtype=float)
    west_north = np.array(
        [[-matrix[0, 0], -matrix[0, 1]], [matrix[1, 0], matrix[1, 1]]]
    )
    lengths = np.linalg.norm(west_north, axis=0)
    if np.any(~np.isfinite(lengths)) or np.any(lengths == 0):
        raise ValueError("The celestial WCS has a singular pixel orientation")
    return west_north / lengths


def load_model(
    data_dir: Path,
    cluster: ClusterSpec,
    model_key: str,
    alignment: str,
) -> RawModel:
    """Load psi and deflection maps, using alpha_x as the canonical WCS.

    Some supplied CATS psi headers have incorrect CRPIX values even though the
    psi and deflection arrays have identical shapes.  The deflection header is
    therefore intentionally used as the shared WCS after shape validation.
    """

    spec = MODEL_SPECS[model_key]
    psi, _ = read_fits_array(product_path(data_dir, cluster, spec, "psi"))
    alpha_x, alpha_header = read_fits_array(
        product_path(data_dir, cluster, spec, "x-arcsec-deflect")
    )
    alpha_y, _ = read_fits_array(
        product_path(data_dir, cluster, spec, "y-arcsec-deflect")
    )
    if psi.shape != alpha_x.shape or alpha_y.shape != alpha_x.shape:
        raise ValueError(
            f"{spec.label} psi/deflection shapes do not match: "
            f"psi={psi.shape}, alpha_x={alpha_x.shape}, alpha_y={alpha_y.shape}"
        )

    wcs = WCS(alpha_header).celestial
    if not wcs.has_celestial:
        raise ValueError(f"No celestial WCS in {product_path(data_dir, cluster, spec, 'x-arcsec-deflect')}")
    wcs, alignment_note = validate_or_recentre_wcs(
        wcs, alpha_header, alpha_x.shape, alignment
    )
    scales = np.asarray(proj_plane_pixel_scales(wcs), dtype=float) * 3600.0
    factor = cluster.deflection_scale
    return RawModel(
        spec=spec,
        psi=psi * factor,
        alpha_x=alpha_x * factor,
        alpha_y=alpha_y * factor,
        wcs=wcs,
        header=alpha_header,
        pixel_scales=(float(scales[0]), float(scales[1])),
        orientation=deflection_orientation(wcs),
        alignment_note=alignment_note,
    )


def validate_or_recentre_wcs(
    wcs: WCS,
    header: fits.Header,
    shape: tuple[int, int],
    alignment: str,
) -> tuple[WCS, str]:
    """Use FITS WCS, or repair a clearly stale cutout reference pixel.

    Several CATS arrays in this working data set were made with ``Cutout2D``
    more than once.  Their scale/orientation remains useful, but in some files
    the target sky position lands far from the array centre or outside it.  In
    ``auto`` mode a displacement exceeding 30 percent of either array dimension
    is treated as a stale CRPIX and the target is placed at the array centre.
    ``wcs`` and ``center`` force the two behaviours explicitly.
    """

    if alignment == "wcs":
        return wcs, "FITS WCS"
    if "RA_TARG" not in header or "DEC_TARG" not in header:
        if alignment == "center":
            raise ValueError("--alignment center requires RA_TARG and DEC_TARG")
        return wcs, "FITS WCS (no target metadata for sanity check)"

    target = SkyCoord(
        float(header["RA_TARG"]) * u.deg,
        float(header["DEC_TARG"]) * u.deg,
        frame="icrs",
    )
    target_x, target_y = wcs.world_to_pixel(target)
    ny, nx = shape
    centre_x = (nx - 1) / 2.0
    centre_y = (ny - 1) / 2.0
    suspicious = (
        not np.isfinite(target_x)
        or not np.isfinite(target_y)
        or abs(float(target_x) - centre_x) > 0.30 * nx
        or abs(float(target_y) - centre_y) > 0.30 * ny
    )
    if alignment == "auto" and not suspicious:
        return wcs, "FITS WCS (target-position sanity check passed)"

    repaired = wcs.deepcopy()
    # WCS CRPIX uses the FITS one-based convention.
    repaired.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    repaired.wcs.crval = [float(header["RA_TARG"]), float(header["DEC_TARG"])]
    repaired.wcs.set()
    reason = "forced centre alignment" if alignment == "center" else "stale CRPIX fallback"
    return repaired, f"target centred ({reason})"


def reference_sky_position(models: list[RawModel]) -> SkyCoord:
    """Choose the common tangent point, preferring the products' target sky position."""

    coordinates: list[tuple[float, float]] = []
    for model in models:
        if "RA_TARG" in model.header and "DEC_TARG" in model.header:
            coordinates.append(
                (float(model.header["RA_TARG"]), float(model.header["DEC_TARG"]))
            )
        else:
            ny, nx = model.alpha_x.shape
            centre = model.wcs.pixel_to_world((nx - 1) / 2, (ny - 1) / 2)
            coordinates.append((float(centre.ra.deg), float(centre.dec.deg)))
    # The models of a given cluster occupy a small field far from the RA wrap.
    return SkyCoord(
        ra=np.mean([item[0] for item in coordinates]) * u.deg,
        dec=np.mean([item[1] for item in coordinates]) * u.deg,
        frame="icrs",
    )


def footprint_bounds(model: RawModel, centre: SkyCoord) -> tuple[float, float, float, float]:
    """Return the west/north bounding box of a model footprint in arcseconds."""

    ny, nx = model.alpha_x.shape
    px = np.array([-0.5, nx - 0.5, nx - 0.5, -0.5])
    py = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5])
    corners = model.wcs.pixel_to_world(px, py)
    offsets = corners.transform_to(SkyOffsetFrame(origin=centre))
    west_values = -offsets.lon.to_value(u.arcsec)
    north_values = offsets.lat.to_value(u.arcsec)
    return (
        float(np.min(west_values)),
        float(np.max(west_values)),
        float(np.min(north_values)),
        float(np.max(north_values)),
    )


def make_common_grid(
    models: list[RawModel],
    centre: SkyCoord,
    minimum_scale: float,
    max_pixels: int | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Construct a common grid over the intersection of model footprints."""

    bounds = [footprint_bounds(model, centre) for model in models]
    x_min = max(item[0] for item in bounds)
    x_max = min(item[1] for item in bounds)
    y_min = max(item[2] for item in bounds)
    y_max = min(item[3] for item in bounds)
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("The selected models have no overlapping sky footprint")

    scale = float(minimum_scale)
    if max_pixels is not None:
        scale = max(
            scale,
            (x_max - x_min) / max_pixels,
            (y_max - y_min) / max_pixels,
        )
    nx = max(3, int(np.floor((x_max - x_min) / scale)))
    ny = max(3, int(np.floor((y_max - y_min) / scale)))
    x = x_min + (np.arange(nx) + 0.5) * scale
    y = y_min + (np.arange(ny) + 0.5) * scale
    return x, y, scale


def target_pixels_on_source(
    source_wcs: WCS,
    centre: SkyCoord,
    x: np.ndarray,
    y: np.ndarray,
    chunk_rows: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Map common west/north grid points to a source array's pixel coordinates."""

    source_x = np.empty((y.size, x.size), dtype=np.float64)
    source_y = np.empty_like(source_x)
    for start in range(0, y.size, chunk_rows):
        stop = min(start + chunk_rows, y.size)
        xx, yy = np.meshgrid(x, y[start:stop])
        sky = SkyCoord(
            lon=-xx.ravel() * u.arcsec,
            lat=yy.ravel() * u.arcsec,
            frame=SkyOffsetFrame(origin=centre),
        ).transform_to(centre.frame)
        px, py = source_wcs.world_to_pixel(sky)
        source_x[start:stop] = np.asarray(px).reshape(xx.shape)
        source_y[start:stop] = np.asarray(py).reshape(xx.shape)
    return source_x, source_y


def sample_array(
    data: np.ndarray,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    native_scales: tuple[float, float],
    output_scale: float,
) -> np.ndarray:
    """Anti-alias and bilinearly sample an array on a new angular grid."""

    ratio_x = output_scale / native_scales[0]
    ratio_y = output_scale / native_scales[1]
    if max(ratio_x, ratio_y) > 1.001:
        # A Gaussian with sigma~half the downsampling ratio suppresses spatial
        # frequencies above the output Nyquist limit.  Without this step the
        # 0.06-arcsec Keeton pixels alias into false small critical curves when
        # compared with the 0.4--0.5-arcsec Diego grids.
        sigma_pixels = (
            0.5 * np.sqrt(max(ratio_y**2 - 1.0, 0.0)),
            0.5 * np.sqrt(max(ratio_x**2 - 1.0, 0.0)),
        )
        if np.all(np.isfinite(data)):
            data = gaussian_filter(data, sigma_pixels, mode="nearest")
        else:
            finite = np.isfinite(data)
            weights = gaussian_filter(finite.astype(np.float32), sigma_pixels)
            values = gaussian_filter(np.where(finite, data, 0.0), sigma_pixels)
            data = np.divide(
                values,
                weights,
                out=np.full_like(values, np.nan),
                where=weights > 1e-6,
            )

    return map_coordinates(
        data,
        [pixel_y, pixel_x],
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )


def reproject_model(
    model: RawModel,
    centre: SkyCoord,
    x: np.ndarray,
    y: np.ndarray,
    output_scale: float,
) -> PreparedModel:
    """Resample scalar/vector maps and rotate alpha into west/north components."""

    pixel_x, pixel_y = target_pixels_on_source(model.wcs, centre, x, y)
    psi = sample_array(
        model.psi, pixel_x, pixel_y, model.pixel_scales, output_scale
    )
    native_x = sample_array(
        model.alpha_x, pixel_x, pixel_y, model.pixel_scales, output_scale
    )
    native_y = sample_array(
        model.alpha_y, pixel_x, pixel_y, model.pixel_scales, output_scale
    )
    alpha_x = model.orientation[0, 0] * native_x + model.orientation[0, 1] * native_y
    alpha_y = model.orientation[1, 0] * native_x + model.orientation[1, 1] * native_y
    return PreparedModel(
        spec=model.spec,
        psi=psi,
        alpha_x=alpha_x,
        alpha_y=alpha_y,
    )


def largest_true_rectangle(mask: np.ndarray) -> tuple[slice, slice]:
    """Find the largest axis-aligned all-True rectangle in a Boolean mask."""

    heights = np.zeros(mask.shape[1], dtype=np.int64)
    best_area = 0
    best = (0, mask.shape[0], 0, mask.shape[1])
    for row_index, row in enumerate(mask):
        heights = np.where(row, heights + 1, 0)
        stack: list[int] = []
        for column in range(mask.shape[1] + 1):
            current = int(heights[column]) if column < mask.shape[1] else 0
            while stack and current < heights[stack[-1]]:
                top = stack.pop()
                height = int(heights[top])
                left = stack[-1] + 1 if stack else 0
                area = height * (column - left)
                if area > best_area:
                    best_area = area
                    best = (row_index - height + 1, row_index + 1, left, column)
            stack.append(column)
    if best_area == 0:
        raise ValueError("No finite common rectangle remains after WCS reprojection")
    row_start, row_stop, col_start, col_stop = best
    return slice(row_start, row_stop), slice(col_start, col_stop)


def crop_to_common_finite_area(
    models: list[PreparedModel], x: np.ndarray, y: np.ndarray
) -> tuple[list[PreparedModel], np.ndarray, np.ndarray]:
    """Crop rotated footprints to their largest fully valid common rectangle."""

    valid = np.ones((y.size, x.size), dtype=bool)
    for model in models:
        valid &= np.isfinite(model.psi)
        valid &= np.isfinite(model.alpha_x)
        valid &= np.isfinite(model.alpha_y)
    row_slice, col_slice = largest_true_rectangle(valid)
    for model in models:
        model.psi = model.psi[row_slice, col_slice]
        model.alpha_x = model.alpha_x[row_slice, col_slice]
        model.alpha_y = model.alpha_y[row_slice, col_slice]
    return models, x[col_slice], y[row_slice]


def remove_affine_potential_gauge(
    psi: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Subtract best-fitting a + b*x + c*y from psi for visualisation only."""

    xx, yy = np.meshgrid(x, y)
    # Fit on a representative subset to keep this cheap for very large maps.
    stride = max(1, int(np.ceil(max(psi.shape) / 500)))
    sampled_psi = psi[::stride, ::stride].ravel()
    sampled_x = xx[::stride, ::stride].ravel()
    sampled_y = yy[::stride, ::stride].ravel()
    good = np.isfinite(sampled_psi)
    design = np.column_stack(
        [np.ones(np.count_nonzero(good)), sampled_x[good], sampled_y[good]]
    )
    coefficients, *_ = np.linalg.lstsq(design, sampled_psi[good], rcond=None)
    plane = coefficients[0] + coefficients[1] * xx + coefficients[2] * yy
    return psi - plane


def calibrate_potential_units(
    model: PreparedModel, x: np.ndarray, y: np.ndarray
) -> None:
    """Scale psi so that its angular gradient matches the supplied alpha maps.

    The model families do not all store psi in the same units (Diego, for
    example, stores angular potential in radian-squared units).  FITS BUNIT is
    absent, so one conversion is inferred from both deflection components.
    Constant component offsets are allowed because they are a linear-potential
    gauge and do not affect lensing observables.
    """

    gradient_y, gradient_x = np.gradient(model.psi, y, x)
    stride = max(1, int(np.ceil(max(model.psi.shape) / 500)))
    gx = gradient_x[1:-1:stride, 1:-1:stride].ravel()
    gy = gradient_y[1:-1:stride, 1:-1:stride].ravel()
    ax = model.alpha_x[1:-1:stride, 1:-1:stride].ravel()
    ay = model.alpha_y[1:-1:stride, 1:-1:stride].ravel()
    good = np.isfinite(gx) & np.isfinite(gy) & np.isfinite(ax) & np.isfinite(ay)
    gx, gy, ax, ay = gx[good], gy[good], ax[good], ay[good]
    if gx.size < 20:
        raise ValueError(f"Too few finite pixels to calibrate {model.spec.label} psi units")

    gx_c, gy_c = gx - np.mean(gx), gy - np.mean(gy)
    ax_c, ay_c = ax - np.mean(ax), ay - np.mean(ay)
    denominator = np.dot(gx_c, gx_c) + np.dot(gy_c, gy_c)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError(f"Degenerate psi gradient for {model.spec.label}")
    potential_scale = float(
        (np.dot(gx_c, ax_c) + np.dot(gy_c, ay_c)) / denominator
    )
    if not np.isfinite(potential_scale) or potential_scale <= 0:
        raise ValueError(
            f"Could not infer a positive psi unit scale for {model.spec.label}: "
            f"{potential_scale}"
        )
    residual = np.concatenate(
        [ax_c - potential_scale * gx_c, ay_c - potential_scale * gy_c]
    )
    target = np.concatenate([ax_c, ay_c])
    target_spread = float(np.sqrt(np.mean(target**2)))
    fit_error = float(np.sqrt(np.mean(residual**2)))
    if target_spread > 0:
        fit_error /= target_spread
    model.psi *= potential_scale
    model.potential_unit_scale = potential_scale
    model.potential_fit_error = fit_error


def derive_caustics(
    model: PreparedModel,
    x: np.ndarray,
    y: np.ndarray,
    source_reference: np.ndarray,
    minimum_critical_length: float,
) -> None:
    """Use lenstronomy to find and ray-trace critical curves into caustics."""

    d_alpha_x_dy, d_alpha_x_dx = np.gradient(model.alpha_x, y, x)
    d_alpha_y_dy, d_alpha_y_dx = np.gradient(model.alpha_y, y, x)

    # INTERPOL accepts one mixed Hessian term.  Averaging the two numerical
    # estimates enforces the physical symmetry d(alpha_x)/dy=d(alpha_y)/dx and
    # suppresses derivative noise without changing the supplied deflections.
    mixed_hessian = 0.5 * (d_alpha_x_dy + d_alpha_y_dx)
    kwargs_lens = [
        {
            "grid_interp_x": x,
            "grid_interp_y": y,
            "f_": model.psi,
            "f_x": model.alpha_x,
            "f_y": model.alpha_y,
            "f_xx": d_alpha_x_dx,
            "f_yy": d_alpha_y_dy,
            "f_xy": mixed_hessian,
        }
    ]
    lens_model = LensModel(lens_model_list=["INTERPOL"])
    extension = LensModelExtensions(lens_model)

    grid_scale = float(max(np.median(np.diff(x)), np.median(np.diff(y))))
    number_of_pixels = min(x.size, y.size)
    compute_window = number_of_pixels * grid_scale * (1.0 + 1e-10)
    center_x = float((x[0] + x[-1]) / 2.0)
    center_y = float((y[0] + y[-1]) / 2.0)
    critical_x, critical_y, caustic_x, caustic_y = (
        extension.critical_curve_caustics(
            kwargs_lens,
            compute_window=compute_window,
            grid_scale=grid_scale,
            center_x=center_x,
            center_y=center_y,
        )
    )

    reference_alpha_x, reference_alpha_y = lens_model.alpha(
        source_reference[0], source_reference[1], kwargs_lens
    )
    model.source_gauge_alpha = np.array(
        [float(reference_alpha_x), float(reference_alpha_y)]
    )
    model.source_reference_theta = source_reference.copy()
    gauge_shift = model.source_gauge_alpha - source_reference
    model.caustics = []
    model.critical_curves = []
    for theta_x, theta_y, beta_x, beta_y in zip(
        critical_x, critical_y, caustic_x, caustic_y
    ):
        critical_length = float(
            np.sum(np.hypot(np.diff(theta_x), np.diff(theta_y)))
        )
        if critical_length < minimum_critical_length:
            continue
        source_curve = np.column_stack(
            [beta_x + gauge_shift[0], beta_y + gauge_shift[1]]
        )
        critical_curve = np.column_stack([theta_x, theta_y])
        source_curve = source_curve[np.all(np.isfinite(source_curve), axis=1)]
        critical_curve = critical_curve[
            np.all(np.isfinite(critical_curve), axis=1)
        ]
        if source_curve.shape[0] >= 5 and critical_curve.shape[0] >= 5:
            model.caustics.append(source_curve)
            model.critical_curves.append(critical_curve)


def source_plane_reference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Choose one common image-plane point that defines beta=(0, 0)."""

    if x[0] <= 0.0 <= x[-1] and y[0] <= 0.0 <= y[-1]:
        return np.array([0.0, 0.0])
    # This only applies when the target is outside the common footprint.  Use
    # one shared point rather than independently choosing a point per model.
    return np.array([(x[0] + x[-1]) / 2.0, (y[0] + y[-1]) / 2.0])


def image_extent(x: np.ndarray, y: np.ndarray) -> list[float]:
    """Convert pixel-centre coordinates to an imshow edge extent."""

    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    return [x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2]


def robust_symmetric_limit(arrays: list[np.ndarray], percentile: float = 99.0) -> float:
    """Return a shared robust symmetric colour limit."""

    values = np.concatenate([np.abs(array[np.isfinite(array)]).ravel() for array in arrays])
    limit = float(np.percentile(values, percentile))
    return limit if np.isfinite(limit) and limit > 0 else 1.0


def shared_curve_limits(
    curves: list[np.ndarray],
    fallback_extent: list[float],
    reference: np.ndarray,
    robust: bool,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Find common plot bounds for curves from every selected model."""

    if not curves:
        return (fallback_extent[0], fallback_extent[1]), (
            fallback_extent[2],
            fallback_extent[3],
        )
    points = np.vstack(curves)
    if robust:
        x_low, x_high = np.percentile(points[:, 0], [0.5, 99.5])
        y_low, y_high = np.percentile(points[:, 1], [0.5, 99.5])
    else:
        x_low, x_high = np.min(points[:, 0]), np.max(points[:, 0])
        y_low, y_high = np.min(points[:, 1]), np.max(points[:, 1])
    x_low = min(float(x_low), float(reference[0]))
    x_high = max(float(x_high), float(reference[0]))
    y_low = min(float(y_low), float(reference[1]))
    y_high = max(float(y_high), float(reference[1]))
    width = max(float(x_high - x_low), 1.0)
    height = max(float(y_high - y_low), 1.0)
    return (
        (float(x_low - 0.08 * width), float(x_high + 0.08 * width)),
        (float(y_low - 0.08 * height), float(y_high + 0.08 * height)),
    )


def plot_comparison(
    cluster: ClusterSpec,
    models: list[PreparedModel],
    x: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Plot potential, source-plane caustic, and image-plane critical rows."""

    display_psi = [remove_affine_potential_gauge(model.psi, x, y) for model in models]
    psi_limit = robust_symmetric_limit(display_psi)

    n_models = len(models)
    figure, axes = plt.subplots(
        3,
        n_models,
        figsize=(5.3 * n_models, 13.2),
        squeeze=False,
        constrained_layout=True,
        sharex="row",
        sharey="row",
    )
    figure.suptitle(
        f"{cluster.label}: lens-model comparison "
        f"($z_s={cluster.source_redshift:.3f}$)",
        fontsize=17,
    )
    extent = image_extent(x, y)
    caustic_curves = [
        curve for model in models for curve in (model.caustics or [])
    ]
    critical_curves = [
        curve for model in models for curve in (model.critical_curves or [])
    ]
    reference = models[0].source_reference_theta
    if reference is None:
        reference = np.array([0.0, 0.0])
    caustic_xlim, caustic_ylim = shared_curve_limits(
        caustic_curves, extent, np.array([0.0, 0.0]), robust=True
    )
    critical_xlim, critical_ylim = shared_curve_limits(
        critical_curves, extent, reference, robust=False
    )
    psi_norm = TwoSlopeNorm(vmin=-psi_limit, vcenter=0.0, vmax=psi_limit)

    psi_image = None
    for column, (model, psi_for_display) in enumerate(zip(models, display_psi)):
        psi_axis, caustic_axis, critical_axis = axes[:, column]
        psi_image = psi_axis.imshow(
            psi_for_display,
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            norm=psi_norm,
            interpolation="nearest",
        )
        psi_axis.set_title(
            f"{model.spec.label}\n" + r"potential $\psi$",
            fontsize=13,
        )
        psi_axis.set_xlim(extent[0], extent[1])
        psi_axis.set_ylim(extent[2], extent[3])

        for curve in model.caustics or []:
            caustic_axis.plot(curve[:, 0], curve[:, 1], color="#d62728", linewidth=1.3)
        caustic_axis.axhline(0.0, color="0.75", linewidth=0.7, linestyle=":")
        caustic_axis.axvline(0.0, color="0.75", linewidth=0.7, linestyle=":")
        caustic_axis.plot(0.0, 0.0, marker="+", color="black", markersize=7)
        caustic_axis.set_xlim(*caustic_xlim)
        caustic_axis.set_ylim(*caustic_ylim)
        caustic_axis.set_title(
            f"{model.spec.label}\n"
            + r"caustics",
            fontsize=13,
        )

        for curve in model.critical_curves or []:
            critical_axis.plot(
                curve[:, 0], curve[:, 1], color="#1f77b4", linewidth=1.3
            )
        critical_axis.axhline(0.0, color="0.75", linewidth=0.7, linestyle=":")
        critical_axis.axvline(0.0, color="0.75", linewidth=0.7, linestyle=":")
        critical_axis.set_xlim(*critical_xlim)
        critical_axis.set_ylim(*critical_ylim)
        critical_axis.set_title(
            f"{model.spec.label}\n" + r"critical curves (image plane)",
            fontsize=13,
        )

        model_reference = model.source_reference_theta
        if model_reference is not None:
            psi_axis.plot(
                model_reference[0],
                model_reference[1],
                marker="+",
                color="#00bcd4",
                markersize=7,
                markeredgewidth=1.1,
            )
            critical_axis.plot(
                model_reference[0],
                model_reference[1],
                marker="+",
                color="#00bcd4",
                markersize=7,
                markeredgewidth=1.1,
            )

        for axis in (psi_axis, caustic_axis, critical_axis):
            axis.set_aspect("equal", adjustable="box")
            axis.tick_params(labelsize=10)
        psi_axis.set_xlabel(r"image-plane $\theta_x$[arcsec]")
        psi_axis.set_ylabel(r"image-plane $\theta_y$ [arcsec]")
        caustic_axis.set_xlabel(r"source-plane $\beta_x$[arcsec]")
        caustic_axis.set_ylabel(r"source-plane $\beta_y$[arcsec]")
        critical_axis.set_xlabel(r"image-plane $\theta_x$[arcsec]")
        critical_axis.set_ylabel(r"image-plane $\theta_y$ [arcsec]")
    if psi_image is not None:
        figure.colorbar(
            psi_image, ax=axes[0, :].tolist(), shrink=0.86, label=r"$\psi$"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    cluster_key = next(key for key, value in CLUSTERS.items() if value == cluster)
    for extension in formats:
        path = output_dir / f"{cluster_key}_model_comparison.{extension}"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        output_paths.append(path)
    plt.close(figure)
    return output_paths


def normalise_model_keys(keys: list[str]) -> list[str]:
    """Resolve convenient aliases and reject unknown/duplicate model names."""

    normalised: list[str] = []
    for key in keys:
        resolved = MODEL_ALIASES.get(key.lower(), key.lower())
        if resolved not in MODEL_SPECS:
            choices = ", ".join(sorted(MODEL_SPECS))
            raise ValueError(f"Unknown model {key!r}; configured models are: {choices}")
        if resolved not in normalised:
            normalised.append(resolved)
    if len(normalised) < 2:
        raise ValueError("Select at least two different models for a comparison")
    return normalised


def compare_cluster(
    cluster_key: str,
    model_keys: list[str],
    data_dir: Path,
    output_dir: Path,
    max_pixels: int,
    alignment: str,
    minimum_critical_length: float,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Load, align, derive, and plot every selected model for one cluster."""

    cluster = CLUSTERS[cluster_key]
    raw_models = [
        load_model(data_dir, cluster, key, alignment) for key in model_keys
    ]
    centre = reference_sky_position(raw_models)
    comparison_scale = max(model.pixel_scale for model in raw_models)
    x, y, actual_scale = make_common_grid(
        raw_models, centre, comparison_scale, max_pixels
    )
    models = [
        reproject_model(model, centre, x, y, actual_scale)
        for model in raw_models
    ]
    models, x, y = crop_to_common_finite_area(models, x, y)
    source_reference = source_plane_reference(x, y)
    for model in models:
        calibrate_potential_units(model, x, y)

    # Derive each model's caustics from its full lens-plane footprint.  Using
    # only the common psi-display intersection can clip a large critical curve
    # and make its mapped caustic resemble an image-plane arc.
    caustic_grid_info: list[tuple[int, int, float]] = []
    for raw_model, display_model in zip(raw_models, models):
        # Keep the full model footprint, but derive every model's critical
        # curves at the same physical angular resolution.  This is essential
        # for a fair comparison when Keeton has 0.06-arcsec pixels and Diego
        # has roughly 0.4--0.5-arcsec pixels.
        full_x, full_y, caustic_scale = make_common_grid(
            [raw_model], centre, comparison_scale, None
        )
        full_model = reproject_model(
            raw_model, centre, full_x, full_y, caustic_scale
        )
        [full_model], full_x, full_y = crop_to_common_finite_area(
            [full_model], full_x, full_y
        )
        if not (
            full_x[0] <= source_reference[0] <= full_x[-1]
            and full_y[0] <= source_reference[1] <= full_y[-1]
        ):
            raise ValueError(
                f"The shared source-plane reference is outside the "
                f"{raw_model.spec.label} lens-map footprint"
            )
        derive_caustics(
            full_model,
            full_x,
            full_y,
            source_reference,
            minimum_critical_length,
        )
        display_model.caustics = full_model.caustics
        display_model.critical_curves = full_model.critical_curves
        display_model.source_gauge_alpha = full_model.source_gauge_alpha
        display_model.source_reference_theta = full_model.source_reference_theta
        caustic_grid_info.append((full_x.size, full_y.size, caustic_scale))

    display_potentials = [
        remove_affine_potential_gauge(model.psi, x, y) for model in models
    ]

    print(f"\n{cluster.label}")
    print(
        f"  common centre: RA={centre.ra.deg:.7f} deg, "
        f"Dec={centre.dec.deg:.7f} deg"
    )
    print(
        f"  common grid: {x.size} x {y.size} pixels at "
        f"{actual_scale:.4f} arcsec/pixel"
    )
    print(
        f"  shared lensing geometry: z_l={cluster.lens_redshift:.3f}, "
        f"z_s={cluster.source_redshift:.6f}, "
        f"D_ls/D_s={cluster.deflection_scale:.9f}"
    )
    print(
        "  image-plane extent: "
        f"x=[{x[0]:.3f}, {x[-1]:.3f}], y=[{y[0]:.3f}, {y[-1]:.3f}] arcsec"
    )
    print(
        "  common source-plane gauge: beta=(0, 0) at "
        f"theta=({source_reference[0]:.3f}, {source_reference[1]:.3f}) arcsec"
    )
    print(
        f"  caustics use lenstronomy INTERPOL on each full lens footprint; "
        f"critical curves shorter than "
        f"{minimum_critical_length:.3g} arcsec are omitted"
    )
    for first in range(len(models)):
        for second in range(first + 1, len(models)):
            correlation = float(
                np.corrcoef(
                    display_potentials[first].ravel(),
                    display_potentials[second].ravel(),
                )[0, 1]
            )
            print(
                f"  common-grid psi correlation "
                f"({models[first].spec.label} vs {models[second].spec.label}): "
                f"r={correlation:.4f}"
            )
    for raw_model, prepared_model, grid_info in zip(
        raw_models, models, caustic_grid_info
    ):
        caustic_nx, caustic_ny, caustic_scale = grid_info
        matrix_text = np.array2string(raw_model.orientation, precision=3)
        native_x_scale, native_y_scale = raw_model.pixel_scales
        print(
            f"  {raw_model.spec.label}: native={raw_model.alpha_x.shape[1]} x "
            f"{raw_model.alpha_x.shape[0]}, pixel scale (x, y)="
            f"({native_x_scale:.4f}, {native_y_scale:.4f}) arcsec, "
            f"native->west/north={matrix_text}"
        )
        print(f"    alignment: {raw_model.alignment_note}")
        print(
            f"    caustic grid: {caustic_nx} x {caustic_ny} at "
            f"{caustic_scale:.4f} arcsec/pixel (shared angular resolution)"
        )
        print(
            f"    inferred psi unit scale: {prepared_model.potential_unit_scale:.7g} "
            f"(relative gradient-fit RMS={prepared_model.potential_fit_error:.3g})"
        )
        print(
            "    alpha at source-plane reference: "
            f"({prepared_model.source_gauge_alpha[0]:.5g}, "
            f"{prepared_model.source_gauge_alpha[1]:.5g}) arcsec"
        )
    return plot_comparison(
        cluster,
        models,
        x,
        y,
        output_dir,
        formats,
        dpi,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare lens potential and source-plane caustics after WCS "
            "alignment onto a common sky grid."
        )
    )
    parser.add_argument(
        "--clusters",
        nargs="+",
        default=["all"],
        choices=["all", *CLUSTERS],
        help="Cluster key(s) to compare; default: all",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["cats", "williams", "diego", "keeton"],
        help="Configured model keys; default: cats williams diego keeton",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SCRIPT_DIR / "GCdata",
        help="Root containing the per-cluster model directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "model_comparisons",
        help="Directory for comparison figures",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1200,
        help="Maximum pixels on either common-grid axis; default: 1200",
    )
    parser.add_argument(
        "--alignment",
        choices=["auto", "wcs", "center"],
        default="auto",
        help=(
            "Position alignment: validate FITS WCS and repair stale cutout CRPIX "
            "values (auto), trust WCS, or centre every target; default: auto"
        ),
    )
    parser.add_argument(
        "--min-critical-length",
        type=float,
        default=20.0,
        help=(
            "Minimum image-plane critical-curve length retained when forming "
            "cluster caustics, in arcsec; default: 20"
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg"],
        help="Output figure format(s); default: png",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Raster DPI; default: 180")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Reject nonsensical numerical settings with clear messages."""

    if args.max_pixels < 20:
        raise ValueError("--max-pixels must be at least 20")
    if args.min_critical_length < 0:
        raise ValueError("--min-critical-length cannot be negative")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")


def main() -> None:
    """Command-line entry point."""

    args = parse_args()
    validate_args(args)
    model_keys = normalise_model_keys(args.models)
    cluster_keys = list(CLUSTERS) if "all" in args.clusters else args.clusters

    print(
        "Percentage-difference maps are intentionally omitted: psi has an "
        "arbitrary affine gauge and caustics require a curve-distance comparison."
    )
    all_outputs: list[Path] = []
    for cluster_key in cluster_keys:
        all_outputs.extend(
            compare_cluster(
                cluster_key=cluster_key,
                model_keys=model_keys,
                data_dir=args.data_dir.resolve(),
                output_dir=args.output_dir.resolve(),
                max_pixels=args.max_pixels,
                alignment=args.alignment,
                minimum_critical_length=args.min_critical_length,
                formats=args.formats,
                dpi=args.dpi,
            )
        )
    print("\nSaved comparison figures:")
    for path in all_outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
