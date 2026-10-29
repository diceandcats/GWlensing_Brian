"""Simulate with CATS, then localize with CATS, Keeton and Sharon.

    python simulation_three_models.py --row 0

Accepted fits are stored as JSON lists
in CATS_results, Keeton_results and Sharon_results, ordered by cluster.
Run status and output paths are stored in the three_model_* columns.
"""

import argparse
import json
import os
from pathlib import Path

# Sets worker thread limits before NumPy is imported.
from cluster_local_tidy import ClusterLensing
from csv_lock import update_csv_row
from lensing_data_class import LensingData
from simulation_diff import (
    CLUSTERS,
    native_geometry,
    potential_unit_scale,
    product_path,
    read_fits_map,
    resample_to_shape,
)

from astropy.cosmology import FlatLambdaCDM
import corner
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ORDER = ("cats", "keeton", "sharon")
MODEL_LABELS = {"cats": "CATS", "keeton": "Keeton", "sharon": "Sharon"}
# Each cluster/model pair gets its own index in the fitter.
CANDIDATES = tuple(
    (cluster_index, model_name)
    for cluster_index in range(len(CLUSTERS))
    for model_name in MODEL_ORDER
)
BURN_IN_STEPS = 3000

OUTPUT_DEFAULTS = {
    "three_model_run_status": "STARTED",
    "three_model_run_msg": "analysis in progress",
    "three_model_accepted_clusters": "[]",
    "three_model_out_dir": "",
    "three_model_results_file": "PENDING",
    "localization_model_order": json.dumps([MODEL_LABELS[name] for name in MODEL_ORDER]),
    "CATS_results": "[]",
    "Keeton_results": "[]",
    "Sharon_results": "[]",
}


def build_lensing_data():
    """Load the deflection, potential and uncertainty maps for all candidates."""
    alpha_x_maps = []
    alpha_y_maps = []
    psi_maps = []
    sigma_dt_maps = []
    pixscales = []
    x_centers = []
    y_centers = []
    search_windows = []
    redshifts = []

    for candidate_index, (cluster_index, model_name) in enumerate(CANDIDATES):
        cluster = CLUSTERS[cluster_index]
        alpha_x, header = read_fits_map(
            product_path(cluster, model_name, "x-arcsec-deflect")
        )
        alpha_y, _ = read_fits_map(
            product_path(cluster, model_name, "y-arcsec-deflect")
        )
        psi, _ = read_fits_map(product_path(cluster, model_name, "psi"))
        sigma_dt, _ = read_fits_map(product_path(cluster, model_name, "sigma_dt"))
        if alpha_x.shape != alpha_y.shape or alpha_x.shape != psi.shape:
            raise ValueError(
                f"deflection/psi map shapes differ for {cluster.key}/{model_name}"
            )

        geometry, geometry_note = native_geometry(
            cluster, model_name, header, alpha_x.shape
        )
        # CATS potentials already use the units expected by the fitter.
        psi_scale = 1.0
        if model_name != "cats":
            psi_scale, fit_error = potential_unit_scale(
                psi, alpha_x, alpha_y, geometry.pixscale
            )
            psi *= np.float32(psi_scale)
            if fit_error > 0.35:
                print(
                    f"Warning: gradient(psi)-alpha relative RMS={fit_error:.3g} "
                    f"for {cluster.key}/{model_name}",
                    flush=True,
                )
        sigma_dt = resample_to_shape(sigma_dt, alpha_x.shape)

        ny, nx = alpha_x.shape
        height = (ny - 1) * geometry.pixscale
        width = (nx - 1) * geometry.pixscale
        if not 0 < geometry.x_center < width or not 0 < geometry.y_center < height:
            raise ValueError(f"search center is outside {cluster.key}/{model_name}")
        search_window = cluster.search_window
        if model_name != "cats":
            half_width = min(geometry.x_center, width - geometry.x_center)
            half_height = min(geometry.y_center, height - geometry.y_center)
            # Leave a 10% margin at the map edges.
            search_window = min(
                search_window, 1.8 * half_width, 1.8 * half_height
            )

        alpha_x_maps.append(alpha_x)
        alpha_y_maps.append(alpha_y)
        psi_maps.append(psi)
        sigma_dt_maps.append(sigma_dt)
        pixscales.append(geometry.pixscale)
        x_centers.append(geometry.x_center)
        y_centers.append(geometry.y_center)
        search_windows.append(search_window)
        redshifts.append(cluster.lens_redshift)
        print(
            f"Candidate {candidate_index}: cluster {cluster_index}, "
            f"{MODEL_LABELS[model_name]}, shape={alpha_x.shape}, "
            f"pixscale={geometry.pixscale:.6g}, "
            f"center=({geometry.x_center:.4g}, {geometry.y_center:.4g}) "
            f"[{geometry_note}], psi_scale={psi_scale:.7g}",
            flush=True,
        )

    return LensingData(
        alpha_maps_x=alpha_x_maps,
        alpha_maps_y=alpha_y_maps,
        lens_potential_maps=psi_maps,
        uncertainty_dt=sigma_dt_maps,
        pixscale=pixscales,
        z_l_list=redshifts,
        x_center=x_centers,
        y_center=y_centers,
        search_window_list=search_windows,
    )


def save_plots(posterior_path, out_dir, stem, real_params, model_name):
    """Save corner and trace plots for one fit."""
    with np.load(posterior_path) as saved:
        flat_chain = saved["flat_chain"]
        full_chain = saved["chain"]
        labels = saved["param_labels"]

    # Other models use different source coordinates.
    truths = None
    if model_name == "cats":
        truths = [real_params[label] for label in labels]
    figure = corner.corner(
        flat_chain,
        labels=labels,
        quantiles=[0.05, 0.5, 0.95],
        show_titles=True,
        truths=truths,
        label_kwargs={"fontsize": 14},
        title_kwargs={"fontsize": 12},
        verbose=False,
    )
    corner_path = out_dir / f"{stem}_corner_from_saved.png"
    figure.savefig(corner_path, dpi=150)
    plt.close(figure)

    n_steps, _, n_dim = full_chain.shape
    figure, axes = plt.subplots(
        n_dim, figsize=(12, 2 * n_dim), sharex=True, squeeze=False
    )
    steps = np.arange(n_steps)
    for i in range(n_dim):
        ax = axes[i, 0]
        ax.plot(steps, full_chain[:, :, i], alpha=0.2)
        ax.set_ylabel(labels[i], fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
    axes[-1, 0].set_xlabel("Step Number", fontsize=14)
    figure.suptitle("Trace Plot for MCMC parameters", fontsize=16, y=0.99)
    figure.tight_layout(rect=[0, 0, 1, 0.98])
    trace_path = out_dir / f"{stem}_trace.png"
    figure.savefig(trace_path, dpi=150)
    plt.close(figure)
    return str(corner_path), str(trace_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=SCRIPT_DIR / "src_pos_tidy_xyz_copy.csv",
        help="input/output CSV with indices,x,y,z and optional H0",
    )
    parser.add_argument("--row", type=int, required=True, help="zero-based source row")
    parser.add_argument(
        "--test",
        action="store_true",
        help="validate maps and simulate the event without fitting or writing results",
    )
    args = parser.parse_args(argv)
    if args.row < 0:
        parser.error("--row must be non-negative")
    return args


def main(argv=None):
    args = parse_args(argv)
    csv_path = args.csv.resolve()
    frame = pd.read_csv(csv_path)
    if args.row >= len(frame):
        raise IndexError(f"row {args.row} outside CSV with {len(frame)} rows")
    row = frame.iloc[args.row]

    real_params = {
        "x_src": float(row["x"]),
        "y_src": float(row["y"]),
        "z_s": 1.5,
        "H0": 70.0,
    }
    if "z" in row.index and pd.notna(row["z"]):
        real_params["z_s"] = float(row["z"])
    if "H0" in row.index and pd.notna(row["H0"]):
        real_params["H0"] = float(row["H0"])
    cluster_value = float(row["indices"])
    if not cluster_value.is_integer() or not 0 <= cluster_value < len(CLUSTERS):
        raise ValueError(f"cluster index must be 0..5, got {cluster_value}")
    real_cluster = int(cluster_value)
    if real_params["z_s"] <= CLUSTERS[real_cluster].lens_redshift:
        raise ValueError("source redshift must exceed the lens redshift")

    base_output_dir = Path(os.environ.get("OUT_DIR", ".")).resolve()
    out_dir = base_output_dir / f"test_tidy_cats_three_models_row{args.row}"
    if not args.test:
        out_dir.mkdir(parents=True, exist_ok=True)
        updates = OUTPUT_DEFAULTS.copy()
        updates["three_model_out_dir"] = str(out_dir)
        update_csv_row(csv_path, args.row, updates)

    try:
        lensing_data = build_lensing_data()
        cluster_system = ClusterLensing(
            data=lensing_data,
            z_s_ref=real_params["z_s"],
            cosmo_H0=real_params["H0"],
        )
        print("Setup complete. All cluster/model candidates initialized.", flush=True)

        simulation_index = CANDIDATES.index((real_cluster, "cats"))
        output = cluster_system.calculate_imgs_delays_magns(real_params, simulation_index)
        dt_true = np.asarray(output["time_delays"], dtype=float)
        if dt_true.size == 0:
            raise RuntimeError(f"CATS produced no images for cluster {real_cluster}: {real_params}")
        print(f"True image positions: {output['image_positions']}")
        print(f"True time delays: {dt_true}")

        cosmos = FlatLambdaCDM(H0=real_params["H0"], Om0=0.3)
        lum_dist_unlensed = cosmos.luminosity_distance(real_params["z_s"]).value
        mu = np.asarray(output["magnifications"], dtype=float)
        # Same luminosity-distance convention as simulation.py.
        lum_dist_true = lum_dist_unlensed / np.abs(mu)
        print(f"True luminosity distance: {lum_dist_unlensed}")
        print(f"Lensed luminosity distances: {lum_dist_true}")
        if args.test:
            print("Preparation complete; DE/MCMC skipped and spreadsheet unchanged.")
            return

        mcmc_settings = {
            "n_walkers": 20,
            "n_steps": 20000,
            "burn_in_steps": BURN_IN_STEPS,
            "fit_z": True,
            "fit_hubble": False,
            "lum_dist_true": lum_dist_true,
            "sigma_lum": 0.1,
            "z_bounds": (1.0, 5.0),
            "H0_bounds": (60, 80),
        }
        print("\nRunning DE for every cluster/model pair, then thresholded MCMC...", flush=True)
        # Compare all 18 candidates using the thresholds from simulation.py.
        mcmc_results, accepted_candidate_indices = cluster_system.find_best_fit(
            dt_true=dt_true,
            run_mcmc=True,
            mcmc_settings=mcmc_settings,
        )
        accepted_clusters = sorted({CANDIDATES[i][0] for i in accepted_candidate_indices})
        print(f"Accepted physical cluster indices: {accepted_clusters}")
        should_plot = accepted_clusters == [real_cluster]
        combined_rows = []
        model_results = {name: [] for name in MODEL_ORDER}

        for result in sorted(mcmc_results, key=lambda item: item["cluster_index"]):
            candidate_index = result["cluster_index"]
            cluster_index, model_name = CANDIDATES[candidate_index]
            model_label = MODEL_LABELS[model_name]
            print(f"\n--- Processing Cluster {cluster_index}, {model_label} ---", flush=True)
            sampler = result["mcmc_sampler"]
            labels = list(result["de_params"])
            flat_samples = sampler.get_chain(discard=BURN_IN_STEPS, flat=True)
            median_params = dict(zip(labels, np.median(flat_samples, axis=0)))
            for i, label in enumerate(labels):
                low, median, high = np.percentile(flat_samples[:, i], [9, 50, 95])
                print(f"{label:>7s} = {median:.3f} +{high - median:.3f} / -{median - low:.3f}")

            stem = f"cluster_{cluster_index}_{model_label}"
            posterior_path = out_dir / f"{stem}_posterior.npz"
            cluster_system.save_mcmc_results(
                sampler=sampler,
                best_result=result,
                n_burn_in=BURN_IN_STEPS,
                output_path=str(posterior_path),
                dt_true=dt_true,
                mcmc_settings=mcmc_settings,
            )
            corner_path, trace_path = "", ""
            if should_plot:
                corner_path, trace_path = save_plots(
                    posterior_path, out_dir, stem, real_params, model_name,
                )

            record = {
                "cluster_index": cluster_index,
                "model": model_label,
                "localized_x": float(median_params["x_src"]),
                "localized_y": float(median_params["y_src"]),
                "localized_z": float(median_params.get("z_s", real_params["z_s"])),
                "localized_H0": float(median_params.get("H0", real_params["H0"])),
                "chi_sq": float(result["chi_sq"]),
                "posterior_file": str(posterior_path),
                "corner_plot": corner_path,
                "trace_plot": trace_path,
            }
            model_results[model_name].append(record)
            combined_rows.append(record)

        results_file = out_dir / "localization_results.csv"
        result_columns = [
            "cluster_index", "model", "localized_x", "localized_y", "localized_z",
            "localized_H0", "chi_sq", "posterior_file", "corner_plot", "trace_plot",
        ]
        pd.DataFrame(combined_rows, columns=result_columns).to_csv(results_file, index=False)
        if not combined_rows:
            status = "NO_GUESS"
        elif len(accepted_clusters) == 1:
            status = "OK"
        else:
            status = "MULTIPLE_CLUSTERS"

        updates = {
            "three_model_run_status": status,
            "three_model_run_msg": (
                f"{len(combined_rows)} accepted model fits in {len(accepted_clusters)} clusters"
            ),
            "three_model_accepted_clusters": json.dumps(accepted_clusters),
            "three_model_results_file": str(results_file),
        }
        for model_name in MODEL_ORDER:
            column = f"{MODEL_LABELS[model_name]}_results"
            updates[column] = json.dumps(model_results[model_name], allow_nan=False)
        update_csv_row(csv_path, args.row, updates)
        print(f"Updated CSV row {args.row}: {status}; results saved to {results_file}", flush=True)
    except Exception as exc:
        if not args.test:
            update_csv_row(csv_path, args.row, {
                "three_model_run_status": "ERROR",
                "three_model_run_msg": f"{type(exc).__name__}: {exc}",
            })
        raise


if __name__ == "__main__":
    main()
