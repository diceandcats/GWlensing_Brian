#!/usr/bin/env python3
"""Build a Sharon v4 best-fit psi map and a direct-ensemble sigma_dt map.

Purpose
-------
The HFF Sharon v4 range products contain ~100 convergence (kappa) and scalar
shear-magnitude (gamma) maps, but no corresponding potential (psi) maps.  This
script constructs a best-fit Sharon potential and estimates the time-delay
uncertainty from the *realization-level Fermat-related quantity* rather than by
propagating sigma_alpha and sigma_psi as independent errors.

Physics choice
--------------
Only kappa is used to construct the potential variations.  Scalar |gamma| is
NOT used to alter psi because it does not contain the shear orientation and the
validation tests show that fitting scalar gamma does not uniquely/accurately
recover the missing realization-dependent harmonic field.  If paired gamma
files are present, this script reports their count only; use the separate
ensemble-variation validator for gamma diagnostics.

The construction is

    psi_best = psi[kappa_best] + h_best,

where h_best is a low-order harmonic correction (constant deflection + constant
external shear) fitted to the official Sharon best-fit deflection maps.

For the range ensemble,

    delta_psi_i = psi[kappa_i] - psi[<kappa>],

with an affine gauge alignment that removes the unobservable realization-wise
constant-potential and constant-deflection modes.  The candidate realization is

    psi_i   = psi_best + delta_psi_i
    alpha_i = alpha_best + grad(delta_psi_i).

At each pixel we then evaluate the complete Fermat-related lens term

    Q_i = 0.5 * |alpha_i|^2 - psi_i,

and calculate its sample standard deviation directly across realizations,

    sigma_Q = std_i(Q_i, ddof=1).

This retains the alpha/psi covariance and the nonlinear |alpha|^2 term.  The
fractional map used by the existing localization pipeline is

    sigma_dt = sigma_Q / |Q_anchor|,

where Q_anchor is built from the official best-fit alpha and the validated
best-fit psi on the range grid.  This matches the existing project's convention
of a *per-pixel relative time-delay uncertainty map*, but replaces the old
independent-error propagation with the direct ensemble scatter.

Important limitation
--------------------
A single 2-D sigma_dt map cannot encode the full covariance of relative time
delays between arbitrary image pairs.  The exact pairwise quantity would be
std_i[Q_i(theta_a) - Q_i(theta_b)].  Therefore this output is the best
physics-consistent drop-in replacement for the current map interface, not a
full covariance likelihood.  Use --save-fermat-stack if you want to retain the
realization-level Q_i fields for later pairwise calculations.

Expected input layout
---------------------
<sharon-dir>/
    hlsp_frontier_model_<cluster>_sharon_v4_kappa.fits
    hlsp_frontier_model_<cluster>_sharon_v4_x-arcsec-deflect.fits
    hlsp_frontier_model_<cluster>_sharon_v4_y-arcsec-deflect.fits
    [hlsp_frontier_model_<cluster>_sharon_v4_gamma*.fits]
    range/
        ...sharon-map000_v4_kappa.fits
        ...sharon-map000_v4_gamma.fits   # optional for this script
        ...

Outputs
-------
The output directory contains, among other diagnostics:

    hlsp_frontier_model_<cluster>_sharon_v4_psi.fits
    hlsp_frontier_model_<cluster>_sharon_v4_sigma_dt.fits
    sigma_dt_range_native.fits
    sigma_fermat_abs_range.fits
    fermat_anchor_range.fits
    summary.txt
    sigma_dt_diagnostics.png

The production sigma_dt product is resampled onto the best-fit Sharon WCS/grid,
so simulation_diff.py does not need to infer the alignment from array shape.

Example
-------
python build_keeton_sigma_dt.py \
    --sharon-dir "GCdata/Abell 2744/sharon"

By default, all products are kept separate from the downloaded model files in
<sharon-dir>/sigma_dt/.  After inspecting the diagnostics, they can optionally
be installed directly into the Sharon model directory:

python build_keeton_sigma_dt.py \
    --sharon-dir "GCdata/Abell 2744/sharon" \
    --install
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import map_coordinates

from lenstronomy.LensModel.convergence_integrals import potential_from_kappa_grid


MODEL = "sharon"
VERSION = "v4"
MAP_ID_RE = re.compile(r"sharon-map(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class HarmonicCorrection:
    bx: float
    by: float
    gamma1: float
    gamma2: float

    def potential(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (
            self.bx * x
            + self.by * y
            + 0.5 * self.gamma1 * (x**2 - y**2)
            + self.gamma2 * x * y
        )

    def alpha(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ax = self.bx + self.gamma1 * x + self.gamma2 * y
        ay = self.by + self.gamma2 * x - self.gamma1 * y
        return ax, ay


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sharon-dir", type=Path, required=True)
    p.add_argument("--range-dir", type=Path, default=None)
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: <sharon-dir>/sigma_dt).",
    )
    p.add_argument(
        "--edge-fraction", type=float, default=0.15,
        help="Trim fraction per edge for validation/summary statistics (default 0.15).",
    )
    p.add_argument(
        "--fit-stride", type=int, default=4,
        help="Pixel stride used to fit the best-fit harmonic correction (default 4).",
    )
    p.add_argument(
        "--pair-tests", type=int, default=500,
        help="Random line-integral pairs for best-fit psi validation (default 500).",
    )
    p.add_argument(
        "--line-samples", type=int, default=128,
        help="Samples per line integral (default 128).",
    )
    p.add_argument(
        "--max-bestfit-dpsi-rms", type=float, default=0.03,
        help="Abort if harmonic best-fit potential-difference RMS exceeds this value.",
    )
    p.add_argument(
        "--denom-floor-fraction", type=float, default=1e-6,
        help=(
            "Numerical floor for |Q_anchor|, as a fraction of its median absolute "
            "value in the central comparison region (default 1e-6)."
        ),
    )
    p.add_argument(
        "--sigma-scale", type=float, default=1.0,
        help="Optional sensitivity rescaling applied only to sigma_dt (default 1).",
    )
    p.add_argument(
        "--max-range", type=int, default=0,
        help="Use only the first N range maps; 0 uses all maps (debug option).",
    )
    p.add_argument(
        "--wcs-chunk-rows", type=int, default=128,
        help="Rows per chunk for WCS resampling (default 128).",
    )
    p.add_argument(
        "--save-fermat-stack", action="store_true",
        help=(
            "Save all gauge-aligned candidate Q_i maps as a float32 .npy stack. "
            "For 100 x 1201 x 1201 this is about 0.58 GB."
        ),
    )
    p.add_argument(
        "--install", action="store_true",
        help="Copy the production psi and sigma_dt FITS files into --sharon-dir.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Allow --install to replace existing production psi/sigma_dt products.",
    )
    p.add_argument("--seed", type=int, default=20260810)
    args = p.parse_args()

    if not 0 <= args.edge_fraction < 0.45:
        p.error("--edge-fraction must be in [0, 0.45)")
    if args.fit_stride < 1:
        p.error("--fit-stride must be >= 1")
    if args.pair_tests < 10:
        p.error("--pair-tests must be >= 10")
    if args.line_samples < 16:
        p.error("--line-samples must be >= 16")
    if args.max_bestfit_dpsi_rms <= 0:
        p.error("--max-bestfit-dpsi-rms must be positive")
    if args.denom_floor_fraction <= 0:
        p.error("--denom-floor-fraction must be positive")
    if args.sigma_scale <= 0:
        p.error("--sigma-scale must be positive")
    if args.wcs_chunk_rows < 1:
        p.error("--wcs-chunk-rows must be >= 1")
    return args


def read_map(path: Path) -> tuple[np.ndarray, fits.Header]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with fits.open(path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header.copy()
    if data.ndim != 2:
        raise ValueError(f"Expected a 2-D FITS map in {path}, got {data.shape}")
    return data, header


def find_unique(directory: Path, suffix: str, required: bool = True) -> Path | None:
    matches = [
        p for p in directory.glob(f"*_{MODEL}_{VERSION}_{suffix}.fits")
        if f"{MODEL}-map" not in p.name.lower()
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches and not required:
        return None
    if not matches:
        raise FileNotFoundError(f"No best-fit *_{suffix}.fits found in {directory}")
    raise RuntimeError(
        f"Expected one best-fit *_{suffix}.fits in {directory}, found {len(matches)}: "
        + ", ".join(p.name for p in matches)
    )


def discover_range_kappas(range_dir: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for path in range_dir.glob("*.fits"):
        lower = path.name.lower()
        if not lower.endswith(f"_{VERSION}_kappa.fits"):
            continue
        match = MAP_ID_RE.search(path.name)
        if match:
            items.append((match.group(1), path))
    items.sort(key=lambda item: int(item[0]))
    if not items:
        raise FileNotFoundError(
            f"No {MODEL}-mapXXX *_{VERSION}_kappa.fits files in {range_dir}"
        )
    return items


def count_paired_gamma(range_dir: Path, kappas: list[tuple[str, Path]]) -> int:
    gamma_ids: set[str] = set()
    for path in range_dir.glob("*.fits"):
        lower = path.name.lower()
        if not lower.endswith(f"_{VERSION}_gamma.fits"):
            continue
        match = MAP_ID_RE.search(path.name)
        if match:
            gamma_ids.add(match.group(1))
    return sum(map_id in gamma_ids for map_id, _ in kappas)


def pixel_scale_arcsec(header: fits.Header) -> float:
    w = WCS(header).celestial
    scales = np.asarray(proj_plane_pixel_scales(w), dtype=float) * 3600.0
    if scales.size != 2 or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Could not determine a valid celestial pixel scale")
    if not np.isclose(scales[0], scales[1], rtol=5e-3):
        raise ValueError(f"Anisotropic pixels are not supported: {scales}")
    return float(np.mean(scales))


def wcs_orientation_deg(header: fits.Header) -> float:
    matrix = np.asarray(WCS(header).celestial.pixel_scale_matrix, dtype=float)
    if matrix.shape != (2, 2) or np.any(~np.isfinite(matrix)):
        raise ValueError("Could not determine celestial WCS orientation")
    return float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))


def angle_difference_deg(a: float, b: float) -> float:
    return float((a - b + 180.0) % 360.0 - 180.0)


def coordinate_grids(shape: tuple[int, int], pixscale: float) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    x = (np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)) * pixscale
    y = (np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)) * pixscale
    return np.meshgrid(x, y)


def edge_mask(shape: tuple[int, int], fraction: float) -> np.ndarray:
    ny, nx = shape
    ey = int(np.floor(ny * fraction))
    ex = int(np.floor(nx * fraction))
    mask = np.zeros(shape, dtype=bool)
    mask[ey:ny-ey if ey else ny, ex:nx-ex if ex else nx] = True
    return mask


def fit_harmonic_correction(
    residual_x: np.ndarray,
    residual_y: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    stride: int,
) -> HarmonicCorrection:
    """Fit constant deflection + constant external shear to alpha residuals."""
    sample = np.zeros_like(mask)
    sample[::stride, ::stride] = True
    sample &= mask
    valid = sample & np.isfinite(residual_x) & np.isfinite(residual_y)
    if np.count_nonzero(valid) < 20:
        raise ValueError("Too few finite pixels to fit harmonic correction")

    xx = x[valid]
    yy = y[valid]
    rx = residual_x[valid]
    ry = residual_y[valid]

    # alpha_x = bx + gamma1*x + gamma2*y
    # alpha_y = by + gamma2*x - gamma1*y
    ax_design = np.column_stack((np.ones_like(xx), np.zeros_like(xx), xx, yy))
    ay_design = np.column_stack((np.zeros_like(xx), np.ones_like(xx), -yy, xx))
    design = np.vstack((ax_design, ay_design))
    target = np.concatenate((rx, ry))
    coeff, *_ = np.linalg.lstsq(design, target, rcond=None)
    bx, by, gamma1, gamma2 = coeff
    return HarmonicCorrection(float(bx), float(by), float(gamma1), float(gamma2))


def vector_relative_rms(
    dx: np.ndarray,
    dy: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    mask: np.ndarray,
) -> float:
    valid = mask & np.isfinite(dx) & np.isfinite(dy) & np.isfinite(ref_x) & np.isfinite(ref_y)
    num = float(np.sqrt(np.mean(dx[valid]**2 + dy[valid]**2)))
    den = float(np.sqrt(np.mean(ref_x[valid]**2 + ref_y[valid]**2)))
    return num / den if den > 0 else np.nan


def sample_map(data: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    return map_coordinates(data, [row, col], order=1, mode="nearest", prefilter=False)


def potential_difference_test(
    alpha_x: np.ndarray,
    alpha_y: np.ndarray,
    psi: np.ndarray,
    mask: np.ndarray,
    pixscale: float,
    n_pairs: int,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Compare psi(B)-psi(A) to integral alpha.dtheta for random point pairs."""
    rows, cols = np.nonzero(mask)
    if rows.size < 2:
        raise ValueError("Too few pixels for potential-difference validation")
    ia = rng.integers(0, rows.size, size=n_pairs)
    ib = rng.integers(0, rows.size, size=n_pairs)
    ra, ca = rows[ia].astype(float), cols[ia].astype(float)
    rb, cb = rows[ib].astype(float), cols[ib].astype(float)

    t = np.linspace(0.0, 1.0, n_samples)
    rr = ra[:, None] + (rb - ra)[:, None] * t[None, :]
    cc = ca[:, None] + (cb - ca)[:, None] * t[None, :]
    ax = sample_map(alpha_x, rr.ravel(), cc.ravel()).reshape(n_pairs, n_samples)
    ay = sample_map(alpha_y, rr.ravel(), cc.ravel()).reshape(n_pairs, n_samples)
    dx = (cb - ca) * pixscale
    dy = (rb - ra) * pixscale
    integrand = ax * dx[:, None] + ay * dy[:, None]
    line = np.trapz(integrand, t, axis=1)

    dpsi = sample_map(psi, rb, cb) - sample_map(psi, ra, ca)
    resid = dpsi - line
    den = float(np.sqrt(np.mean(line**2)))
    rel_rms = float(np.sqrt(np.mean(resid**2)) / den) if den > 0 else np.nan
    corr = float(np.corrcoef(dpsi, line)[0, 1]) if n_pairs > 1 else np.nan
    med = float(np.median(np.abs(resid)))
    p95 = float(np.percentile(np.abs(resid), 95))
    return rel_rms, corr, med, p95


def resample_wcs(
    source: np.ndarray,
    source_header: fits.Header,
    target_header: fits.Header,
    target_shape: tuple[int, int],
    chunk_rows: int,
    *,
    mode: str = "constant",
) -> np.ndarray:
    """Bilinearly sample a scalar/image-component map from one celestial WCS to another."""
    sw = WCS(source_header).celestial
    tw = WCS(target_header).celestial
    if not sw.has_celestial or not tw.has_celestial:
        raise ValueError("Both source and target headers require celestial WCS")

    ny, nx = target_shape
    out = np.empty(target_shape, dtype=np.float64)
    xpix = np.arange(nx, dtype=np.float64)
    cval = np.nan if mode == "constant" else 0.0
    for y0 in range(0, ny, chunk_rows):
        y1 = min(ny, y0 + chunk_rows)
        yy, xx = np.meshgrid(np.arange(y0, y1, dtype=np.float64), xpix, indexing="ij")
        lon, lat = tw.pixel_to_world_values(xx, yy)
        sx, sy = sw.world_to_pixel_values(lon, lat)
        vals = map_coordinates(
            source,
            [sy.ravel(), sx.ravel()],
            order=1,
            mode=mode,
            cval=cval,
            prefilter=False,
        )
        out[y0:y1] = vals.reshape(y1-y0, nx)
    return out


def ensure_same_component_orientation(source_header: fits.Header, target_header: fits.Header) -> float:
    src = wcs_orientation_deg(source_header)
    dst = wcs_orientation_deg(target_header)
    delta = angle_difference_deg(dst, src)
    if abs(delta) > 1e-3:
        raise ValueError(
            "Best-fit and range WCS axes are rotated by "
            f"{delta:.6g} deg. alpha_x/alpha_y are vector components; this script "
            "will not silently resample them without an explicit basis rotation."
        )
    return delta


def affine_gauge_align_delta(
    delta_psi: np.ndarray,
    pixscale: float,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Remove constant-potential and constant-deflection gauge modes.

    The reference is the central pixel.  We remove c + bx*x + by*y from delta_psi
    and the corresponding constant (bx, by) from grad(delta_psi).  This leaves
    kappa/shear variations unchanged and prevents unobservable affine gauge modes
    from contributing to the ensemble scatter.
    """
    dpsi_dy, dpsi_dx = np.gradient(delta_psi, pixscale, pixscale)
    iy = delta_psi.shape[0] // 2
    ix = delta_psi.shape[1] // 2
    c = float(delta_psi[iy, ix])
    bx = float(dpsi_dx[iy, ix])
    by = float(dpsi_dy[iy, ix])
    x0 = float(x[iy, ix])
    y0 = float(y[iy, ix])

    aligned_psi = delta_psi - c - bx * (x - x0) - by * (y - y0)
    aligned_ax = dpsi_dx - bx
    aligned_ay = dpsi_dy - by
    return aligned_psi, aligned_ax, aligned_ay, c, bx, by


def write_fits(path: Path, data: np.ndarray, header: fits.Header, bunit: str, history: list[str]) -> None:
    h = header.copy()
    h["BUNIT"] = bunit
    for item in history:
        h.add_history(item)
    fits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=h).writeto(path, overwrite=True)


def save_diagnostic_plot(
    path: Path,
    psi_best: np.ndarray,
    q_anchor_range: np.ndarray,
    sigma_q_range: np.ndarray,
    sigma_dt_range: np.ndarray,
    mask_range: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    items = [
        (psi_best, "Validated best-fit Sharon psi [arcsec^2]"),
        (q_anchor_range, "Q_anchor = 0.5|alpha|^2 - psi [arcsec^2]"),
        (sigma_q_range, "Direct ensemble sigma_Q [arcsec^2]"),
        (np.where(mask_range, sigma_dt_range, np.nan), "Fractional sigma_dt (central region)"),
    ]
    for ax, (data, title) in zip(axes.ravel(), items):
        im = ax.imshow(data, origin="lower")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def install_product(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to replace existing production product {dst}. "
            "Use --overwrite only after inspecting the diagnostics."
        )
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    sharon_dir = args.sharon_dir.resolve()
    range_dir = (args.range_dir or (sharon_dir / "range")).resolve()
    out = (args.output_dir or (sharon_dir / "sigma_dt")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not sharon_dir.is_dir():
        raise FileNotFoundError(f"Sharon directory does not exist: {sharon_dir}")
    if not range_dir.is_dir():
        raise FileNotFoundError(f"Range directory does not exist: {range_dir}")

    kappa_path = find_unique(sharon_dir, "kappa")
    alpha_x_path = find_unique(sharon_dir, "x-arcsec-deflect")
    alpha_y_path = find_unique(sharon_dir, "y-arcsec-deflect")

    kappa_best, best_header = read_map(kappa_path)
    alpha_x_best, ax_header = read_map(alpha_x_path)
    alpha_y_best, ay_header = read_map(alpha_y_path)
    if alpha_x_best.shape != kappa_best.shape or alpha_y_best.shape != kappa_best.shape:
        raise ValueError(
            f"Best-fit shape mismatch: kappa={kappa_best.shape}, "
            f"alpha_x={alpha_x_best.shape}, alpha_y={alpha_y_best.shape}"
        )

    pix_best = pixel_scale_arcsec(best_header)
    best_mask = edge_mask(kappa_best.shape, args.edge_fraction)
    x_best, y_best = coordinate_grids(kappa_best.shape, pix_best)

    print("=== BEST-FIT SHARON V4 POTENTIAL ===")
    print(f"kappa: {kappa_path.name}")
    print(f"shape: {kappa_best.shape}")
    print(f"pixel scale: {pix_best:.8g} arcsec/pixel")

    # One FFT potential reconstruction; derive alpha from the same psi so the
    # Fermat construction is internally consistent and memory-efficient.
    psi_raw = np.asarray(potential_from_kappa_grid(kappa_best, pix_best), dtype=np.float64)
    alpha_y_raw, alpha_x_raw = np.gradient(psi_raw, pix_best, pix_best)

    correction = fit_harmonic_correction(
        alpha_x_best-alpha_x_raw,
        alpha_y_best-alpha_y_raw,
        x_best,
        y_best,
        best_mask,
        args.fit_stride,
    )
    corr_ax, corr_ay = correction.alpha(x_best, y_best)
    psi_best = psi_raw + correction.potential(x_best, y_best)
    alpha_x_rec = alpha_x_raw + corr_ax
    alpha_y_rec = alpha_y_raw + corr_ay

    alpha_rms = vector_relative_rms(
        alpha_x_rec-alpha_x_best,
        alpha_y_rec-alpha_y_best,
        alpha_x_best,
        alpha_y_best,
        best_mask,
    )
    dpsi_rms, dpsi_corr, dpsi_med, dpsi_p95 = potential_difference_test(
        alpha_x_best,
        alpha_y_best,
        psi_best,
        best_mask,
        pix_best,
        args.pair_tests,
        args.line_samples,
        rng,
    )

    print(
        "harmonic correction: "
        f"bx={correction.bx:+.6g}, by={correction.by:+.6g}, "
        f"gamma1={correction.gamma1:+.6g}, gamma2={correction.gamma2:+.6g}"
    )
    print(f"alpha relative RMS after harmonic correction: {alpha_rms:.6g}")
    print(f"potential-difference relative RMS: {dpsi_rms:.6g}")
    print(f"potential-difference correlation: {dpsi_corr:.8g}")

    if not np.isfinite(dpsi_rms) or dpsi_rms > args.max_bestfit_dpsi_rms:
        raise RuntimeError(
            "Best-fit kappa->psi anchor failed the configured time-delay-relevant "
            f"validation: dpsi relative RMS={dpsi_rms:.6g} > "
            f"{args.max_bestfit_dpsi_rms:.6g}. Refusing to build sigma_dt."
        )

    product_stem = kappa_path.name[:-len("_kappa.fits")]
    psi_name = product_stem + "_psi.fits"
    sigma_name = product_stem + "_sigma_dt.fits"
    psi_out = out / psi_name
    sigma_out = out / sigma_name

    write_fits(
        psi_out,
        psi_best,
        best_header,
        "arcsec2",
        [
            "Sharon psi reconstructed from best-fit kappa using lenstronomy FFT Green-function convolution.",
            "Low-order harmonic correction fitted to official Sharon x/y deflection maps.",
            f"Best-fit potential-difference validation relative RMS={dpsi_rms:.6g}.",
        ],
    )

    # ------------------------------------------------------------------
    # Range pass 1: establish common native grid and mean kappa.
    # ------------------------------------------------------------------
    kappas = discover_range_kappas(range_dir)
    if args.max_range:
        kappas = kappas[:args.max_range]
    if len(kappas) < 2:
        raise ValueError("Need at least two kappa realizations")
    paired_gamma_count = count_paired_gamma(range_dir, kappas)

    mean_kappa = None
    range_header = None
    range_shape = None
    pix_range = None

    print("\n=== RANGE ENSEMBLE ===")
    print(f"kappa realizations: {len(kappas)}")
    print(f"paired scalar-gamma maps found: {paired_gamma_count}")
    print("scalar gamma usage: QC/provenance only; NOT used to construct psi or sigma_dt")

    for j, (map_id, path) in enumerate(kappas, start=1):
        kappa_i, header_i = read_map(path)
        pix_i = pixel_scale_arcsec(header_i)
        if range_shape is None:
            range_shape = kappa_i.shape
            range_header = header_i
            pix_range = pix_i
            mean_kappa = np.zeros(range_shape, dtype=np.float64)
            print(f"range shape: {range_shape}")
            print(f"range pixel scale: {pix_range:.8g} arcsec/pixel")
        else:
            if kappa_i.shape != range_shape:
                raise ValueError(
                    f"Range map {map_id} shape {kappa_i.shape} differs from {range_shape}"
                )
            if not np.isclose(pix_i, pix_range, rtol=5e-4):
                raise ValueError(
                    f"Range map {map_id} pixel scale {pix_i} differs from {pix_range}"
                )
        mean_kappa += kappa_i
        if j == 1 or j % 20 == 0 or j == len(kappas):
            print(f"  pass 1: {j:3d}/{len(kappas)}")

    assert mean_kappa is not None and range_header is not None
    assert range_shape is not None and pix_range is not None
    mean_kappa /= float(len(kappas))

    orient_delta = ensure_same_component_orientation(best_header, range_header)
    print(f"best-fit/range WCS orientation difference: {orient_delta:.6g} deg")

    x_range, y_range = coordinate_grids(range_shape, pix_range)
    range_mask = edge_mask(range_shape, args.edge_fraction)

    # Mean raw range potential.  By linearity, psi(kappa_i)-psi(mean_kappa)
    # represents the potential variation implied by delta-kappa on this grid.
    print("Reconstructing mean range potential...")
    psi_mean_raw = np.asarray(
        potential_from_kappa_grid(mean_kappa, pix_range), dtype=np.float64
    )

    # Resample independently validated best-fit anchors to the native range grid.
    psi_anchor = resample_wcs(
        psi_best, best_header, range_header, range_shape, args.wcs_chunk_rows
    )
    alpha_x_anchor = resample_wcs(
        alpha_x_best, ax_header, range_header, range_shape, args.wcs_chunk_rows
    )
    alpha_y_anchor = resample_wcs(
        alpha_y_best, ay_header, range_header, range_shape, args.wcs_chunk_rows
    )

    finite_anchor = (
        np.isfinite(psi_anchor)
        & np.isfinite(alpha_x_anchor)
        & np.isfinite(alpha_y_anchor)
    )
    if np.count_nonzero(range_mask & finite_anchor) < 100:
        raise ValueError("Too few overlapping finite pixels between best-fit and range grids")
    if np.any(~finite_anchor[range_mask]):
        raise ValueError(
            "Best-fit maps do not completely cover the central range comparison region."
        )

    q_anchor = 0.5 * (alpha_x_anchor**2 + alpha_y_anchor**2) - psi_anchor

    # ------------------------------------------------------------------
    # Range pass 2: direct Q_i scatter, with affine gauge alignment.
    # ------------------------------------------------------------------
    q_mean = np.zeros(range_shape, dtype=np.float64)
    q_m2 = np.zeros(range_shape, dtype=np.float64)
    count = 0
    gauge_c: list[float] = []
    gauge_bx: list[float] = []
    gauge_by: list[float] = []

    fermat_stack = None
    if args.save_fermat_stack:
        stack_path = out / "fermat_Q_realizations.npy"
        fermat_stack = np.lib.format.open_memmap(
            stack_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(kappas), range_shape[0], range_shape[1]),
        )
        print(f"Writing realization-level Q_i stack to {stack_path}")

    for j, (map_id, path) in enumerate(kappas, start=1):
        kappa_i, _ = read_map(path)
        psi_i_raw = np.asarray(
            potential_from_kappa_grid(kappa_i, pix_range), dtype=np.float64
        )
        delta_psi_raw = psi_i_raw - psi_mean_raw
        delta_psi, delta_ax, delta_ay, c, bx, by = affine_gauge_align_delta(
            delta_psi_raw, pix_range, x_range, y_range
        )

        psi_i = psi_anchor + delta_psi
        alpha_x_i = alpha_x_anchor + delta_ax
        alpha_y_i = alpha_y_anchor + delta_ay
        q_i = 0.5 * (alpha_x_i**2 + alpha_y_i**2) - psi_i

        count += 1
        delta = q_i - q_mean
        q_mean += delta / count
        q_m2 += delta * (q_i - q_mean)

        gauge_c.append(c)
        gauge_bx.append(bx)
        gauge_by.append(by)

        if fermat_stack is not None:
            fermat_stack[j-1] = q_i.astype(np.float32)

        if j == 1 or j % 10 == 0 or j == len(kappas):
            print(
                f"  pass 2: {j:3d}/{len(kappas)} map{map_id} "
                f"gauge(c,bx,by)=({c:+.4g},{bx:+.4g},{by:+.4g})"
            )

    if count < 2:
        raise RuntimeError("Need at least two realizations for sample variance")
    sigma_q = np.sqrt(np.maximum(q_m2 / (count - 1), 0.0))

    # Drop-in fractional convention used by the current project.  This is now
    # direct ensemble scatter of Q, not propagated independent alpha/psi errors.
    valid_scale = range_mask & finite_anchor & np.isfinite(q_anchor)
    abs_q_anchor = np.abs(q_anchor)
    robust_q_scale = float(np.median(abs_q_anchor[valid_scale]))
    if not np.isfinite(robust_q_scale) or robust_q_scale <= 0:
        raise RuntimeError("Could not determine a positive robust |Q_anchor| scale")
    denom_floor = args.denom_floor_fraction * robust_q_scale
    denominator = np.maximum(abs_q_anchor, denom_floor)
    sigma_dt_range = args.sigma_scale * sigma_q / denominator

    floor_mask = valid_scale & (abs_q_anchor < denom_floor)
    floor_fraction = float(np.count_nonzero(floor_mask) / np.count_nonzero(valid_scale))

    # Save native-grid diagnostics before any resampling.
    write_fits(
        out / "fermat_anchor_range.fits",
        q_anchor,
        range_header,
        "arcsec2",
        ["Q_anchor = 0.5*|alpha_best|^2 - psi_best on native range grid."],
    )
    write_fits(
        out / "fermat_mean_range.fits",
        q_mean,
        range_header,
        "arcsec2",
        ["Mean of gauge-aligned candidate realization Q_i fields."],
    )
    write_fits(
        out / "sigma_fermat_abs_range.fits",
        sigma_q,
        range_header,
        "arcsec2",
        ["Sample std (ddof=1) of realization-level Q_i = 0.5|alpha_i|^2 - psi_i."],
    )
    write_fits(
        out / "sigma_dt_range_native.fits",
        sigma_dt_range,
        range_header,
        "dimensionless",
        [
            "Fractional time-delay uncertainty: std_i(Q_i)/|Q_anchor|.",
            "Q_i evaluated per realization; alpha/psi covariance retained.",
            "Affine gauge modes removed from each delta-psi realization.",
            f"sigma_scale={args.sigma_scale:g}; denominator floor fraction={args.denom_floor_fraction:g}.",
        ],
    )

    # Resample the scalar uncertainty to the best-fit grid used by simulation_diff.
    sigma_dt_bestgrid = resample_wcs(
        sigma_dt_range,
        range_header,
        best_header,
        kappa_best.shape,
        args.wcs_chunk_rows,
        mode="nearest",
    )
    write_fits(
        sigma_out,
        sigma_dt_bestgrid,
        best_header,
        "dimensionless",
        [
            "Sharon fractional time-delay uncertainty from direct realization-level Fermat scatter.",
            "Constructed from range kappa maps; scalar gamma is not used to modify psi.",
            "Candidate fields anchored to official best-fit alpha and validated best-fit psi.",
            "Affine gauge modes removed from each delta-kappa potential realization.",
            f"NREAL={count}; sigma_scale={args.sigma_scale:g}.",
        ],
    )

    save_diagnostic_plot(
        out / "sigma_dt_diagnostics.png",
        psi_best,
        q_anchor,
        sigma_q,
        sigma_dt_range,
        range_mask & finite_anchor,
    )

    central = range_mask & finite_anchor & np.isfinite(sigma_dt_range)
    sigma_vals = sigma_dt_range[central]
    sigma_q_vals = sigma_q[central]
    q_diff = q_mean - q_anchor
    q_mean_rel_rms = float(
        np.sqrt(np.mean(q_diff[central]**2))
        / max(np.sqrt(np.mean(q_anchor[central]**2)), 1e-30)
    )

    summary = [
        "Sharon v4 direct-Fermat sigma_dt construction",
        "==========================================",
        f"sharon_dir: {sharon_dir}",
        f"range_dir: {range_dir}",
        f"range_realizations: {count}",
        f"paired_scalar_gamma_maps_found: {paired_gamma_count}",
        "gamma_usage: diagnostic/provenance only; NOT used to construct psi or sigma_dt",
        "",
        "Best-fit potential anchor",
        f"bestfit_shape: {kappa_best.shape}",
        f"bestfit_pixscale_arcsec: {pix_best:.10g}",
        f"harmonic_bx_arcsec: {correction.bx:.10g}",
        f"harmonic_by_arcsec: {correction.by:.10g}",
        f"harmonic_gamma1: {correction.gamma1:.10g}",
        f"harmonic_gamma2: {correction.gamma2:.10g}",
        f"alpha_relative_rms_after_harmonic: {alpha_rms:.10g}",
        f"potential_difference_relative_rms: {dpsi_rms:.10g}",
        f"potential_difference_correlation: {dpsi_corr:.10g}",
        f"potential_difference_median_abs_arcsec2: {dpsi_med:.10g}",
        f"potential_difference_p95_abs_arcsec2: {dpsi_p95:.10g}",
        "",
        "Range grid / direct Fermat ensemble",
        f"range_shape: {range_shape}",
        f"range_pixscale_arcsec: {pix_range:.10g}",
        f"bestfit_range_wcs_orientation_delta_deg: {orient_delta:.10g}",
        "construction: psi_i = psi_best_anchor + gauge_aligned[psi(kappa_i)-psi(<kappa>)]",
        "construction: alpha_i = alpha_best_anchor + gauge_aligned grad(delta_psi_i)",
        "construction: Q_i = 0.5*|alpha_i|^2 - psi_i",
        "construction: sigma_Q = sample_std_i(Q_i, ddof=1)",
        "construction: sigma_dt = sigma_scale*sigma_Q/|Q_anchor|",
        f"sigma_scale: {args.sigma_scale:.10g}",
        f"denominator_floor_fraction: {args.denom_floor_fraction:.10g}",
        f"denominator_floor_arcsec2: {denom_floor:.10g}",
        f"central_pixels_using_denominator_floor_fraction: {floor_fraction:.10g}",
        f"mean_Q_vs_anchor_relative_rms: {q_mean_rel_rms:.10g}",
        "",
        "Fractional sigma_dt statistics in central comparison region",
        f"median: {np.median(sigma_vals):.10g}",
        f"p16: {np.percentile(sigma_vals,16):.10g}",
        f"p84: {np.percentile(sigma_vals,84):.10g}",
        f"p95: {np.percentile(sigma_vals,95):.10g}",
        f"p99: {np.percentile(sigma_vals,99):.10g}",
        f"max: {np.max(sigma_vals):.10g}",
        "",
        "Absolute sigma_Q [arcsec^2] statistics in central comparison region",
        f"median: {np.median(sigma_q_vals):.10g}",
        f"p84: {np.percentile(sigma_q_vals,84):.10g}",
        f"p95: {np.percentile(sigma_q_vals,95):.10g}",
        "",
        "Affine gauge alignment across realizations",
        f"removed_constant_potential_std_arcsec2: {np.std(gauge_c, ddof=1):.10g}",
        f"removed_constant_alpha_x_std_arcsec: {np.std(gauge_bx, ddof=1):.10g}",
        f"removed_constant_alpha_y_std_arcsec: {np.std(gauge_by, ddof=1):.10g}",
        "",
        "Interpretation",
        "- This sigma_dt uses the complete realization-level Q_i scatter and therefore retains alpha/psi covariance.",
        "- Scalar gamma is not used because |gamma| alone cannot uniquely determine the shear orientation or lens potential.",
        "- The affine gauge alignment removes unobservable constant-potential and constant-deflection variation before taking the ensemble scatter.",
        "- The output sigma_dt is a per-pixel fractional uncertainty compatible with the current localization-map interface.",
        "- A 2-D map still cannot encode the full covariance of arbitrary relative-delay image pairs; use the optional Q_i stack for exact fixed-position pairwise scatter.",
    ]
    (out / "summary.txt").write_text("\n".join(summary) + "\n")

    print("\n=== OUTPUT ===")
    print(f"best-fit psi: {psi_out}")
    print(f"sigma_dt:     {sigma_out}")
    print(f"summary:      {out / 'summary.txt'}")
    print(f"diagnostics:  {out / 'sigma_dt_diagnostics.png'}")
    print(
        "central sigma_dt: "
        f"median={np.median(sigma_vals):.4g}, "
        f"p84={np.percentile(sigma_vals,84):.4g}, "
        f"p95={np.percentile(sigma_vals,95):.4g}"
    )

    if args.install:
        dst_psi = sharon_dir / psi_name
        dst_sigma = sharon_dir / sigma_name
        install_product(psi_out, dst_psi, args.overwrite)
        install_product(sigma_out, dst_sigma, args.overwrite)
        print("\nInstalled production products:")
        print(f"  {dst_psi}")
        print(f"  {dst_sigma}")


if __name__ == "__main__":
    main()
