"""Simulate an event with one HFF lens model and localize it with another.

Examples
--------
Simulate with CATS and localize with Keeton (the defaults)::

    python simulation_diff.py --csv src_pos_tidy_xyz.csv --row 0

Choose the two families explicitly::

    python simulation_diff.py --csv src_pos_tidy_xyz.csv --row 0 \
        --simulation-model cats --localization-model keeton
"""

# pylint: skip-file
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import warnings

from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import corner
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates

from cluster_local_tidy import ClusterLensing
from csv_lock import update_csv_row
from lensing_data_class import LensingData


warnings.filterwarnings(
    "ignore", category=UserWarning, module="lenstronomy.LensModel.lens_model"
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "GCdata"


@dataclass(frozen=True)
class ClusterSpec:
    key: str
    directory: str
    lens_redshift: float
    search_window: float


@dataclass(frozen=True)
class ModelSpec:
    directory: str
    filename_token: str
    version: str


@dataclass(frozen=True)
class MapGeometry:
    """Map pixel scale and center."""

    pixscale: float
    x_center: float
    y_center: float


CLUSTERS = (
    ClusterSpec("abell370", "Abell 370", 0.375, 90.0),
    ClusterSpec("abell2744", "Abell 2744", 0.308, 120.0),
    ClusterSpec("abells1063", "Abell S1063", 0.351, 100.0),
    ClusterSpec("macs0416", "MACS J0416.1-2403", 0.397, 100.0),
    ClusterSpec("macs0717", "MACS J0717.5+3745", 0.545, 150.0),
    ClusterSpec("macs1149", "MACS J1149.5+2223", 0.543, 100.0),
)

MODELS = {
    "cats": ModelSpec("cats copy", "cats", "v4"),
    "williams": ModelSpec("william", "williams", "v4"),
    "diego": ModelSpec("diego", "diego", "v4"),
    "keeton": ModelSpec("keeton", "keeton", "v4"),
}

# Preserve the CATS coordinates used by the source catalog and simulation.py.
MODEL_GEOMETRY_OVERRIDES = {
    ("cats", "abell370"): MapGeometry(0.2, 77.0, 79.0),
    ("cats", "abell2744"): MapGeometry(0.3, 175.0, 180.0),
    ("cats", "abells1063"): MapGeometry(0.2, 100.0, 100.0),
    ("cats", "macs0416"): MapGeometry(0.3, 120.0, 120.0),
    ("cats", "macs0717"): MapGeometry(0.8, 250.0, 240.0),
    ("cats", "macs1149"): MapGeometry(0.5, 150.0, 150.0),
}


def normalize_model_name(value: str) -> str:
    """Resolve convenient aliases used by the on-disk directories."""

    key = "cats" if value.lower() == "cat" else value.lower()
    if key not in MODELS:
        raise argparse.ArgumentTypeError(
            f"unknown model {value!r}; choose from {', '.join(MODELS)}"
        )
    return key


def product_path(cluster: ClusterSpec, model_name: str, kind: str) -> Path:
    model = MODELS[model_name]
    filename = (
        f"hlsp_frontier_model_{cluster.key}_{model.filename_token}_"
        f"{model.version}_{kind}.fits"
    )
    return DATA_DIR / cluster.directory / model.directory / filename


def read_fits_map(path: Path) -> tuple[np.ndarray, fits.Header]:
    if not path.is_file():
        raise FileNotFoundError(f"missing lens-model product: {path}")
    with fits.open(path, memmap=True) as hdul:
        data = np.array(hdul[0].data, dtype=np.float32, copy=True)
        header = hdul[0].header.copy()
    if data.ndim != 2:
        raise ValueError(f"expected a 2-D FITS image in {path}, got {data.shape}")
    return data, header


def wcs_pixscale(cluster: ClusterSpec, model_name: str, wcs: WCS) -> float:
    """Return an isotropic native pixel scale in arcsec/pixel."""

    scales = np.asarray(proj_plane_pixel_scales(wcs), dtype=float) * 3600.0
    if scales.size != 2 or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError(f"invalid celestial pixel scale for {cluster.key}/{model_name}")
    if not np.isclose(scales[0], scales[1], rtol=5e-3):
        raise ValueError(
            f"anisotropic pixels are not supported for {cluster.key}/{model_name}: "
            f"{scales[0]:.6g} x {scales[1]:.6g} arcsec"
        )
    return float(np.mean(scales))


def native_geometry(
    cluster: ClusterSpec,
    model_name: str,
    header: fits.Header,
    shape: tuple[int, int],
) -> tuple[MapGeometry, str]:
    """Read model-specific scale and target center, with optional overrides."""

    override = MODEL_GEOMETRY_OVERRIDES.get((model_name, cluster.key))
    if override is not None:
        return override, "calibrated override"

    wcs = WCS(header).celestial
    if not wcs.has_celestial:
        raise ValueError(f"missing celestial WCS for {cluster.key}/{model_name}")
    pixscale = wcs_pixscale(cluster, model_name, wcs)
    if "RA_TARG" not in header or "DEC_TARG" not in header:
        raise ValueError(
            f"missing RA_TARG/DEC_TARG needed to center {cluster.key}/{model_name}"
        )

    center_pixel = np.asarray(
        wcs.all_world2pix(
            [[float(header["RA_TARG"]), float(header["DEC_TARG"])]], 0
        )[0],
        dtype=float,
    )
    ny, nx = shape
    if (
        center_pixel.shape != (2,)
        or np.any(~np.isfinite(center_pixel))
        or not 0.0 <= center_pixel[0] <= nx - 1
        or not 0.0 <= center_pixel[1] <= ny - 1
    ):
        raise ValueError(
            f"target center lies outside {cluster.key}/{model_name} map: "
            f"pixel={center_pixel}, shape={shape}"
        )

    return (
        MapGeometry(
            pixscale=pixscale,
            x_center=float(center_pixel[0] * pixscale),
            y_center=float(center_pixel[1] * pixscale),
        ),
        "FITS WCS target",
    )


def potential_unit_scale(
    psi: np.ndarray, alpha_x: np.ndarray, alpha_y: np.ndarray, pixscale: float
) -> tuple[float, float]:
    """Infer the positive scalar that converts psi to angular-potential units."""

    grad_y, grad_x = np.gradient(psi, pixscale, pixscale)
    stride = max(1, int(np.ceil(max(psi.shape) / 500)))
    region = np.s_[1:-1:stride, 1:-1:stride]
    gx = grad_x[region].ravel().astype(np.float64)
    gy = grad_y[region].ravel().astype(np.float64)
    ax = alpha_x[region].ravel().astype(np.float64)
    ay = alpha_y[region].ravel().astype(np.float64)
    finite = np.isfinite(gx) & np.isfinite(gy) & np.isfinite(ax) & np.isfinite(ay)
    gx, gy, ax, ay = gx[finite], gy[finite], ax[finite], ay[finite]
    if gx.size < 20:
        raise ValueError("too few finite map pixels to determine potential units")

    # Ignore constant deflection offsets when fitting the potential units.
    gx -= np.mean(gx)
    gy -= np.mean(gy)
    ax -= np.mean(ax)
    ay -= np.mean(ay)
    denominator = np.dot(gx, gx) + np.dot(gy, gy)
    scale = float((np.dot(gx, ax) + np.dot(gy, ay)) / denominator)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"could not infer a positive potential scale (got {scale})")

    residual = np.concatenate((ax - scale * gx, ay - scale * gy))
    target = np.concatenate((ax, ay))
    target_rms = float(np.sqrt(np.mean(target**2)))
    relative_error = float(np.sqrt(np.mean(residual**2)) / target_rms)
    return scale, relative_error


def resample_to_shape(data: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinearly resample an aligned scalar map to a requested (ny, nx)."""

    if data.shape == shape:
        return np.asarray(data, dtype=np.float32)
    source_y = np.linspace(0.0, data.shape[0] - 1.0, shape[0])
    source_x = np.linspace(0.0, data.shape[1] - 1.0, shape[1])
    xx, yy = np.meshgrid(source_x, source_y)
    return map_coordinates(
        data,
        [yy, xx],
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32)


def limit_map_resolution(
    arrays: tuple[np.ndarray, ...], pixscale: float, max_map_pixels: int
) -> tuple[tuple[np.ndarray, ...], float]:
    """Anti-alias and downsample very large maps while preserving their extent."""

    shape = arrays[0].shape
    if max_map_pixels <= 0 or max(shape) <= max_map_pixels:
        return arrays, pixscale
    ratio = (max(shape) - 1) / (max_map_pixels - 1)
    new_shape = tuple(
        max(3, int(np.floor((size - 1) / ratio + 1e-9)) + 1)
        for size in shape
    )
    source_y = np.arange(new_shape[0], dtype=float) * ratio
    source_x = np.arange(new_shape[1], dtype=float) * ratio
    xx, yy = np.meshgrid(source_x, source_y)
    blur_sigma = 0.5 * np.sqrt(max(ratio**2 - 1.0, 0.0))
    result = []
    for data in arrays:
        smoothed = gaussian_filter(data, blur_sigma, mode="nearest")
        result.append(
            map_coordinates(
                smoothed,
                [yy, xx],
                order=1,
                mode="nearest",
                prefilter=False,
            ).astype(np.float32)
        )
    return tuple(result), pixscale * ratio


def load_cluster_maps(
    cluster: ClusterSpec,
    model_name: str,
    sigma_fallback: float,
    max_map_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Load and validate one cluster/model family on its native grid."""

    alpha_x, header = read_fits_map(product_path(cluster, model_name, "x-arcsec-deflect"))
    alpha_y, _ = read_fits_map(product_path(cluster, model_name, "y-arcsec-deflect"))
    psi, _ = read_fits_map(product_path(cluster, model_name, "psi"))
    if alpha_x.shape != alpha_y.shape or alpha_x.shape != psi.shape:
        raise ValueError(
            f"map shapes differ for {cluster.key}/{model_name}: "
            f"alpha_x={alpha_x.shape}, alpha_y={alpha_y.shape}, psi={psi.shape}"
        )

    geometry, geometry_note = native_geometry(
        cluster, model_name, header, alpha_x.shape
    )
    native_pixscale = geometry.pixscale
    inferred_psi_scale, fit_error = potential_unit_scale(
        psi, alpha_x, alpha_y, native_pixscale
    )
    # Keep the CATS potential unchanged to match simulation.py.
    psi_scale = 1.0 if model_name == "cats" else inferred_psi_scale
    psi *= np.float32(psi_scale)
    native_shape = alpha_x.shape
    (alpha_x, alpha_y, psi), effective_pixscale = limit_map_resolution(
        (alpha_x, alpha_y, psi), native_pixscale, max_map_pixels
    )

    sigma_path = product_path(cluster, model_name, "sigma_dt")
    if sigma_path.is_file():
        sigma_dt, _ = read_fits_map(sigma_path)
        sigma_dt = resample_to_shape(sigma_dt, native_shape)
        if native_shape != alpha_x.shape:
            # Downsample uncertainty on the lens-map grid.
            (sigma_dt,), _ = limit_map_resolution(
                (sigma_dt,), native_pixscale, max_map_pixels
            )
        sigma_note = sigma_path.name
    else:
        sigma_dt = np.full(alpha_x.shape, sigma_fallback, dtype=np.float32)
        sigma_note = f"constant {sigma_fallback:g}"

    print(
        f"[{model_name:8s}] {cluster.key:11s}: shape={alpha_x.shape}, "
        f"pixscale={effective_pixscale:.6g} arcsec, "
        f"center=({geometry.x_center:.4g}, {geometry.y_center:.4g}) arcsec "
        f"[{geometry_note}], psi_scale={psi_scale:.7g}, "
        f"gradient_error={fit_error:.3g}, sigma_dt={sigma_note}",
        flush=True,
    )
    if fit_error > 0.35:
        warnings.warn(
            f"large gradient(psi)-alpha mismatch for {cluster.key}/{model_name}: "
            f"relative RMS={fit_error:.3g}"
        )
    return (
        alpha_x,
        alpha_y,
        psi,
        sigma_dt,
        effective_pixscale,
        geometry.x_center,
        geometry.y_center,
    )


def build_lensing_data(
    model_name: str, sigma_fallback: float, max_map_pixels: int = 1600
) -> LensingData:
    alpha_x_maps = []
    alpha_y_maps = []
    psi_maps = []
    sigma_maps = []
    pixscales = []
    x_centers = []
    y_centers = []
    search_windows = []

    for cluster in CLUSTERS:
        ax, ay, psi, sigma, pix, x_center, y_center = load_cluster_maps(
            cluster, model_name, sigma_fallback, max_map_pixels
        )
        alpha_x_maps.append(ax)
        alpha_y_maps.append(ay)
        psi_maps.append(psi)
        sigma_maps.append(sigma)
        pixscales.append(pix)
        x_centers.append(x_center)
        y_centers.append(y_center)
        height = (ax.shape[0] - 1) * pix
        width = (ax.shape[1] - 1) * pix
        if not 0.0 <= x_center <= width or not 0.0 <= y_center <= height:
            raise ValueError(
                f"center ({x_center:.6g}, {y_center:.6g}) is outside the "
                f"downsampled {cluster.key}/{model_name} extent "
                f"({width:.6g}, {height:.6g}) arcsec"
            )
        centered_width = 2.0 * min(x_center, width - x_center)
        centered_height = 2.0 * min(y_center, height - y_center)
        search_windows.append(
            min(cluster.search_window, 0.9 * centered_width, 0.9 * centered_height)
        )

    return LensingData(
        alpha_maps_x=alpha_x_maps,
        alpha_maps_y=alpha_y_maps,
        lens_potential_maps=psi_maps,
        uncertainty_dt=sigma_maps,
        pixscale=pixscales,
        z_l_list=[item.lens_redshift for item in CLUSTERS],
        x_center=x_centers,
        y_center=y_centers,
        search_window_list=search_windows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="CSV with indices,x,y,z,H0,...")
    parser.add_argument("--row", type=int, required=True, help="zero-based CSV row")
    parser.add_argument(
        "--simulation-model", type=normalize_model_name, default="cats",
        help="model used to generate the event (default: cats)",
    )
    parser.add_argument(
        "--localization-model", type=normalize_model_name, default="keeton",
        help="model used for DE/MCMC localization (default: keeton)",
    )
    parser.add_argument(
        "--sigma-dt-fallback", type=float, default=0.1,
        help="fractional time-delay uncertainty if a model has no sigma_dt FITS map",
    )
    parser.add_argument("--burn-in", type=int, default=3000)
    parser.add_argument(
        "--max-map-pixels", type=int, default=1600,
        help="anti-aliased maximum map dimension; 0 keeps native resolution",
    )
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="load/validate maps and simulate the event, but skip DE/MCMC",
    )
    args = parser.parse_args()
    if args.row < 0:
        parser.error("--row must be non-negative")
    if args.sigma_dt_fallback <= 0:
        parser.error("--sigma-dt-fallback must be positive")
    if args.burn_in >= 20000:
        parser.error("--burn-in must be less than 20000")
    if args.max_map_pixels != 0 and args.max_map_pixels < 32:
        parser.error("--max-map-pixels must be 0 or at least 32")
    return args


OUTPUT_DEFAULTS = {
    "run_status": "STARTED",
    "run_msg": "",
    "localized_index": np.nan,
    "localized_x": np.nan,
    "localized_y": np.nan,
    "localized_z": np.nan,
    "localized_H0": np.nan,
    "chi_sq": np.nan,
    "accepted_clusters": "",
    "out_dir": "",
    "posterior_file": "",
    "corner_plot": "",
    "trace_plot": "",
    "simulation_model": "",
    "localization_model": "",
}


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    frame = pd.read_csv(csv_path)
    row = frame.iloc[args.row]

    real_params = {"x_src": float(row["x"]), "y_src": float(row["y"])}
    if "z" in row.index and pd.notna(row["z"]):
        real_params["z_s"] = float(row["z"])
    if "H0" in row.index and pd.notna(row["H0"]):
        real_params["H0"] = float(row["H0"])
    real_params.setdefault("z_s", 1.5)
    real_cluster = int(row["indices"])

    base_output_dir = Path(os.environ.get("OUT_DIR", ".")).resolve()
    out_dir = base_output_dir / (
        f"test_tidy_{args.simulation_model}_to_{args.localization_model}_row{args.row}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    start_update = {
        **OUTPUT_DEFAULTS,
        "out_dir": str(out_dir),
        "simulation_model": args.simulation_model,
        "localization_model": args.localization_model,
    }
    update_csv_row(csv_path, args.row, start_update)

    try:
        print(f"Loading simulation model: {args.simulation_model}", flush=True)
        simulation_data = build_lensing_data(
            args.simulation_model, args.sigma_dt_fallback, args.max_map_pixels
        )
        if args.localization_model == args.simulation_model:
            localization_data = simulation_data
        else:
            print(f"Loading localization model: {args.localization_model}", flush=True)
            localization_data = build_lensing_data(
                args.localization_model, args.sigma_dt_fallback, args.max_map_pixels
            )

        z_s_ref = real_params.get("z_s", 1.5)
        fixed_h0 = real_params.get("H0", 70.0)
        simulation_system = ClusterLensing(
            data=simulation_data, z_s_ref=z_s_ref, cosmo_H0=fixed_h0
        )
        if localization_data is simulation_data:
            localization_system = simulation_system
        else:
            localization_system = ClusterLensing(
                data=localization_data, z_s_ref=z_s_ref, cosmo_H0=fixed_h0
            )
        print("Setup complete. Both lensing systems are initialized.", flush=True)

        event = simulation_system.calculate_imgs_delays_magns(real_params, real_cluster)
        dt_true = np.asarray(event["time_delays"], dtype=float)
        if dt_true.size == 0:
            raise RuntimeError(
                f"the {args.simulation_model} event produced no images for {real_params}"
            )
        print(f"True image positions: {event['image_positions']}")
        print(f"True time delays (arrival-time order): {dt_true}")

        cosmos = FlatLambdaCDM(H0=real_params.get("H0", 70.0), Om0=0.3)
        lum_dist_unlensed = cosmos.luminosity_distance(real_params["z_s"]).value
        magnifications = np.asarray(event["magnifications"], dtype=float)
        # Match simulation.py's luminosity-distance convention.
        lum_dist_true = lum_dist_unlensed / np.abs(magnifications)
        print(f"True luminosity distance: {lum_dist_unlensed}")
        print(f"Lensed luminosity distances: {lum_dist_true}")

        if args.prepare_only:
            update_csv_row(
                csv_path,
                args.row,
                {
                    "run_status": "PREPARED",
                    "run_msg": f"simulated {dt_true.size} images; localization skipped",
                },
            )
            print("--prepare-only requested; skipping DE and MCMC.")
            return

        mcmc_settings = {
            "n_walkers": 20,
            "n_steps": 20000,
            "burn_in_steps": args.burn_in,
            "fit_z": False,
            "fit_hubble": False,
            "lum_dist_true": lum_dist_true,
            "sigma_lum": 0.1,
            "z_bounds": (1.0, 5.0),
            "H0_bounds": (60, 80),
        }
        print("\nRunning cross-model DE and MCMC localization...")
        mcmc_results, accepted_cluster_indices = localization_system.find_best_fit(
            dt_true=dt_true, run_mcmc=True, mcmc_settings=mcmc_settings
        )
        print(f"Accepted cluster indices: {accepted_cluster_indices}")

        if not mcmc_results:
            update_csv_row(
                csv_path,
                args.row,
                {
                    "accepted_clusters": "N/A",
                    "chi_sq": "N/A",
                    "out_dir": "N/A",
                    "run_status": "NO_GUESS",
                    "run_msg": "len(mcmc_results)=0",
                    "localized_index": "N/A",
                    "localized_x": "N/A",
                    "localized_y": "N/A",
                    "localized_z": "N/A",
                    "localized_H0": "N/A",
                },
            )
            return

        for result in mcmc_results:
            cluster_idx = result["cluster_index"]
            output_path = out_dir / f"cluster_{cluster_idx}_posterior.npz"
            localization_system.save_mcmc_results(
                sampler=result["mcmc_sampler"],
                best_result=result,
                n_burn_in=args.burn_in,
                output_path=str(output_path),
                dt_true=dt_true,
                mcmc_settings=mcmc_settings,
            )

        accepted_str = ",".join(str(item) for item in accepted_cluster_indices)
        base_updates = {"accepted_clusters": accepted_str, "out_dir": str(out_dir)}
        if len(mcmc_results) == 1:
            result = mcmc_results[0]
            cluster_idx = result["cluster_index"]
            labels = list(result["de_params"].keys())
            flat_samples = result["mcmc_sampler"].get_chain(
                discard=args.burn_in, flat=True
            )
            medians = [
                float(np.percentile(flat_samples[:, i], 50))
                for i in range(len(labels))
            ]
            median_params = dict(zip(labels, medians))
            chi_sq_median = localization_system._calculate_chi_squared(
                params=median_params,
                dt_true=dt_true,
                index=cluster_idx,
                sigma_lum=0.1,
                lum_dist_true=lum_dist_true,
            )
            update_csv_row(
                csv_path,
                args.row,
                {
                    **base_updates,
                    "run_status": "OK",
                    "run_msg": "",
                    "localized_index": cluster_idx,
                    "localized_x": median_params["x_src"],
                    "localized_y": median_params["y_src"],
                    "localized_z": median_params.get("z_s", 0.0),
                    "localized_H0": median_params.get("H0", 0.0),
                    "chi_sq": chi_sq_median,
                },
            )
        else:
            update_csv_row(
                csv_path,
                args.row,
                {
                    **base_updates,
                    "run_status": "NO_FIT",
                    "run_msg": f"len(mcmc_results)={len(mcmc_results)}",
                    "localized_index": 0,
                    "localized_x": 0.0,
                    "localized_y": 0.0,
                    "localized_z": 0.0,
                    "localized_H0": 0.0,
                },
            )

        should_plot = (
            len(accepted_cluster_indices) == 1
            and accepted_cluster_indices[0] == real_cluster
        )
        if should_plot:
            posterior_path = out_dir / f"cluster_{real_cluster}_posterior.npz"
            if posterior_path.is_file():
                saved = np.load(posterior_path)
                flat_chain = saved["flat_chain"]
                full_chain = saved["chain"]
                param_labels = saved["param_labels"]
                # Cross-model source coordinates may have different gauges.
                truths = None
                if args.simulation_model == args.localization_model:
                    truths = [real_params["x_src"], real_params["y_src"]]
                    if "z_s" in param_labels and "z_s" in real_params:
                        truths.append(real_params["z_s"])
                    if "H0" in param_labels and "H0" in real_params:
                        truths.append(real_params["H0"])

                figure = corner.corner(
                    flat_chain,
                    labels=param_labels,
                    quantiles=[0.05, 0.5, 0.95],
                    show_titles=True,
                    truths=truths,
                    label_kwargs={"fontsize": 14},
                    title_kwargs={"fontsize": 12},
                    verbose=False,
                )
                corner_path = out_dir / f"cluster_{real_cluster}_corner_from_saved.png"
                figure.savefig(corner_path, dpi=150)
                plt.close(figure)

                n_steps, _, n_dim = full_chain.shape
                trace_figure, axes = plt.subplots(
                    n_dim, figsize=(12, 2 * n_dim), sharex=True, squeeze=False
                )
                steps = np.arange(n_steps)
                for dimension in range(n_dim):
                    axis = axes[dimension, 0]
                    axis.plot(steps, full_chain[:, :, dimension], alpha=0.2)
                    axis.set_ylabel(param_labels[dimension], fontsize=14)
                axes[-1, 0].set_xlabel("Step Number", fontsize=14)
                trace_figure.tight_layout()
                trace_path = out_dir / f"cluster_{real_cluster}_trace.png"
                trace_figure.savefig(trace_path, dpi=150)
                plt.close(trace_figure)
                update_csv_row(
                    csv_path,
                    args.row,
                    {
                        "posterior_file": str(posterior_path),
                        "corner_plot": str(corner_path),
                        "trace_plot": str(trace_path),
                    },
                )
        else:
            print("Plotting skipped: localization did not uniquely select the true cluster.")
    except Exception as exc:
        update_csv_row(
            csv_path,
            args.row,
            {"run_status": "ERROR", "run_msg": f"{type(exc).__name__}: {exc}"},
        )
        raise


if __name__ == "__main__":
    main()
