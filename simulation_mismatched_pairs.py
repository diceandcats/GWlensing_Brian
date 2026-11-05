"""Inject with one lens model and localize with each of the other two.

Example:
    python simulation_mismatched_pairs.py --simulation-model cats \
        --csv src_pos_tidy_xyz_cats.csv --cluster 0
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path

from astropy.cosmology import FlatLambdaCDM
import corner
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cluster_local_tidy import ClusterLensing
from csv_lock import update_csv_row
from simulation_diff import CLUSTERS, build_lensing_data


MODEL_ORDER = ("cats", "keeton", "sharon")
SIGMA_DT_FALLBACK = 0.1
MAX_MAP_PIXELS = 1600
BURN_IN_STEPS = 3000
N_WALKERS = 16
N_STEPS = 20000

RESULT_COLUMNS = [
    "injection_model",
    "localization_model",
    "source_row",
    "injected_cluster",
    "candidate_cluster",
    "localized_x_q09",
    "localized_x_median",
    "localized_x_q95",
    "localized_y_q09",
    "localized_y_median",
    "localized_y_q95",
    "localized_z_q09",
    "localized_z_median",
    "localized_z_q95",
    "localized_H0_q09",
    "localized_H0_median",
    "localized_H0_q95",
    "chi_sq",
    "posterior_file",
    "corner_plot",
    "trace_plot",
]

SUMMARY_COLUMNS = [
    "injection_model",
    "localization_model",
    "source_csv",
    "source_row",
    "injected_cluster",
    "status",
    "message",
    "accepted_clusters",
    "event_file",
    "results_file",
    "out_dir",
]


@dataclass(frozen=True)
class Source:
    csv_path: Path
    row_index: int
    cluster_index: int
    params: dict[str, float]


@dataclass(frozen=True)
class InjectedEvent:
    source: Source
    dt_true: np.ndarray
    lum_dist_true: np.ndarray
    event_file: Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulation-model",
        choices=MODEL_ORDER,
        required=True,
        help="lens-model family used to inject the event",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="source-position CSV for --simulation-model",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--row",
        type=int,
        help="zero-based row in the injection CSV",
    )
    selection.add_argument(
        "--cluster",
        type=int,
        help="physical cluster 0..5; use its first matching injection row",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="inject the event and initialize both recovery models, but skip DE/MCMC",
    )
    args = parser.parse_args(argv)
    if args.row is not None and args.row < 0:
        parser.error("--row must be non-negative")
    if args.cluster is not None and not 0 <= args.cluster < len(CLUSTERS):
        parser.error(f"--cluster must be 0..{len(CLUSTERS) - 1}")
    return args


def read_source(csv_path: Path, row_index: int) -> Source:
    csv_path = csv_path.resolve()
    frame = pd.read_csv(csv_path)
    if row_index >= len(frame):
        raise IndexError(f"row {row_index} outside {csv_path} with {len(frame)} rows")
    row = frame.iloc[row_index]
    missing = {"indices", "x", "y"} - set(row.index)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {', '.join(sorted(missing))}")

    cluster_value = float(row["indices"])
    if not np.isfinite(cluster_value) or not cluster_value.is_integer():
        raise ValueError(f"cluster index must be an integer, got {row['indices']!r}")
    cluster_index = int(cluster_value)
    if not 0 <= cluster_index < len(CLUSTERS):
        raise ValueError(f"cluster index must be 0..{len(CLUSTERS) - 1}, got {cluster_index}")

    params = {
        "x_src": float(row["x"]),
        "y_src": float(row["y"]),
        "z_s": 1.5,
        "H0": 70.0,
    }
    if "z" in row.index and pd.notna(row["z"]):
        params["z_s"] = float(row["z"])
    if "H0" in row.index and pd.notna(row["H0"]):
        params["H0"] = float(row["H0"])
    if not all(np.isfinite(value) for value in params.values()):
        raise ValueError(f"non-finite source parameters in row {row_index} of {csv_path}")
    if params["z_s"] <= CLUSTERS[cluster_index].lens_redshift:
        raise ValueError("source redshift must exceed the injected cluster redshift")
    if params["H0"] <= 0:
        raise ValueError("H0 must be positive")
    return Source(csv_path, row_index, cluster_index, params)


def read_cluster_source(csv_path: Path, cluster_index: int) -> Source:
    csv_path = csv_path.resolve()
    frame = pd.read_csv(csv_path)
    if "indices" not in frame:
        raise ValueError(f"{csv_path} is missing column: indices")
    indices = pd.to_numeric(frame["indices"], errors="coerce").to_numpy()
    matches = np.flatnonzero(indices == cluster_index)
    if matches.size == 0:
        raise ValueError(f"{csv_path} has no source for cluster {cluster_index}")
    return read_source(csv_path, int(matches[0]))


def pair_prefix(injection_model: str, localization_model: str) -> str:
    return f"{injection_model}_to_{localization_model}"


def update_source(source: Source, injection_model: str, localization_model: str, **values):
    prefix = pair_prefix(injection_model, localization_model)
    update_csv_row(
        source.csv_path,
        source.row_index,
        {f"{prefix}_{name}": value for name, value in values.items()},
    )


def simulate_event(model_name: str, source: Source, event_dir: Path) -> InjectedEvent:
    print(f"\n=== Injecting with {model_name.upper()} ===", flush=True)
    data = build_lensing_data(model_name, SIGMA_DT_FALLBACK, MAX_MAP_PIXELS)
    system = ClusterLensing(
        data=data,
        z_s_ref=source.params["z_s"],
        cosmo_H0=source.params["H0"],
    )
    output = system.calculate_imgs_delays_magns(source.params, source.cluster_index)
    dt_true = np.asarray(output["time_delays"], dtype=float)
    magnifications = np.asarray(output.get("magnifications", []), dtype=float)
    if (
        dt_true.size == 0
        or magnifications.size != dt_true.size
        or not np.all(np.isfinite(dt_true))
        or not np.all(np.isfinite(magnifications))
        or np.any(magnifications == 0)
    ):
        raise RuntimeError(
            f"{model_name} produced no usable event for cluster {source.cluster_index}"
        )

    cosmology = FlatLambdaCDM(H0=source.params["H0"], Om0=0.3)
    unlensed_distance = cosmology.luminosity_distance(source.params["z_s"]).value
    lum_dist_true = unlensed_distance / np.abs(magnifications)
    event_file = event_dir / f"{model_name}_event.npz"
    np.savez_compressed(
        event_file,
        injection_model=model_name,
        source_row=source.row_index,
        cluster_index=source.cluster_index,
        x_src=source.params["x_src"],
        y_src=source.params["y_src"],
        z_s=source.params["z_s"],
        H0=source.params["H0"],
        image_x=np.asarray(output["image_positions"][0], dtype=float),
        image_y=np.asarray(output["image_positions"][1], dtype=float),
        time_delays=dt_true,
        magnifications=magnifications,
        luminosity_distances=lum_dist_true,
    )
    print(f"Injected cluster {source.cluster_index}; time delays: {dt_true}", flush=True)
    del system, data
    gc.collect()
    return InjectedEvent(source, dt_true, lum_dist_true, event_file)


def save_plots(result, pair_dir: Path, cluster_index: int):
    labels = list(result["de_params"])
    sampler = result["mcmc_sampler"]
    flat_samples = sampler.get_chain(discard=BURN_IN_STEPS, flat=True)
    figure = corner.corner(
        flat_samples,
        labels=labels,
        quantiles=[0.05, 0.5, 0.95],
        show_titles=True,
        truths=None,
        verbose=False,
    )
    corner_path = pair_dir / f"cluster_{cluster_index}_corner.png"
    figure.savefig(corner_path, dpi=150)
    plt.close(figure)

    chain = sampler.get_chain(discard=BURN_IN_STEPS)
    n_steps, _, n_dim = chain.shape
    figure, axes = plt.subplots(n_dim, figsize=(12, 2 * n_dim), sharex=True, squeeze=False)
    steps = np.arange(n_steps)
    for dimension in range(n_dim):
        axes[dimension, 0].plot(steps, chain[:, :, dimension], alpha=0.2)
        axes[dimension, 0].set_ylabel(labels[dimension])
    axes[-1, 0].set_xlabel("Step Number")
    figure.tight_layout()
    trace_path = pair_dir / f"cluster_{cluster_index}_trace.png"
    figure.savefig(trace_path, dpi=150)
    plt.close(figure)
    return str(corner_path), str(trace_path)


def interval(flat_samples, labels, name, fixed_value):
    if name not in labels:
        value = float(fixed_value)
        return value, value, value
    index = labels.index(name)
    return tuple(float(value) for value in np.percentile(flat_samples[:, index], [9, 50, 95]))


def run_pair(
    injection_model: str,
    localization_model: str,
    event: InjectedEvent,
    localization_data,
    pair_dir: Path,
):
    pair_dir.mkdir(parents=True, exist_ok=True)
    system = ClusterLensing(
        data=localization_data,
        z_s_ref=event.source.params["z_s"],
        cosmo_H0=event.source.params["H0"],
    )
    mcmc_settings = {
        "n_walkers": N_WALKERS,
        "n_steps": N_STEPS,
        "burn_in_steps": BURN_IN_STEPS,
        "autocorr_check_interval": 500,
        "min_steps": 5000,
        "min_autocorr_times": 50,
        "autocorr_rel_tol": 0.01,
        "fit_z": True,
        "fit_hubble": False,
        "lum_dist_true": event.lum_dist_true,
        "sigma_lum": 0.1,
        "z_bounds": (1.0, 5.0),
        "H0_bounds": (60, 80),
    }
    print(
        f"\n=== Localizing {injection_model.upper()} injection with "
        f"{localization_model.upper()} over {len(CLUSTERS)} clusters ===",
        flush=True,
    )
    results, accepted_clusters = system.find_best_fit(
        dt_true=event.dt_true,
        run_mcmc=True,
        mcmc_settings=mcmc_settings,
    )

    rows = []
    unique_match = accepted_clusters == [event.source.cluster_index]
    for result in sorted(results, key=lambda item: item["cluster_index"]):
        cluster_index = int(result["cluster_index"])
        sampler = result["mcmc_sampler"]
        labels = list(result["de_params"])
        flat_samples = sampler.get_chain(discard=BURN_IN_STEPS, flat=True)
        x_interval = interval(flat_samples, labels, "x_src", event.source.params["x_src"])
        y_interval = interval(flat_samples, labels, "y_src", event.source.params["y_src"])
        z_interval = interval(flat_samples, labels, "z_s", event.source.params["z_s"])
        h0_interval = interval(flat_samples, labels, "H0", event.source.params["H0"])

        posterior_path = pair_dir / f"cluster_{cluster_index}_posterior.npz"
        system.save_mcmc_results(
            sampler=sampler,
            best_result=result,
            n_burn_in=BURN_IN_STEPS,
            output_path=str(posterior_path),
            dt_true=event.dt_true,
            mcmc_settings=mcmc_settings,
        )
        corner_path, trace_path = "", ""
        if unique_match and cluster_index == event.source.cluster_index:
            try:
                corner_path, trace_path = save_plots(result, pair_dir, cluster_index)
            except Exception as exc:
                print(f"Plotting skipped: {type(exc).__name__}: {exc}", flush=True)

        rows.append(
            {
                "injection_model": injection_model,
                "localization_model": localization_model,
                "source_row": event.source.row_index,
                "injected_cluster": event.source.cluster_index,
                "candidate_cluster": cluster_index,
                "localized_x_q09": x_interval[0],
                "localized_x_median": x_interval[1],
                "localized_x_q95": x_interval[2],
                "localized_y_q09": y_interval[0],
                "localized_y_median": y_interval[1],
                "localized_y_q95": y_interval[2],
                "localized_z_q09": z_interval[0],
                "localized_z_median": z_interval[1],
                "localized_z_q95": z_interval[2],
                "localized_H0_q09": h0_interval[0],
                "localized_H0_median": h0_interval[1],
                "localized_H0_q95": h0_interval[2],
                "chi_sq": float(result["chi_sq"]),
                "posterior_file": str(posterior_path),
                "corner_plot": corner_path,
                "trace_plot": trace_path,
            }
        )

    results_file = pair_dir / "localization_results.csv"
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(results_file, index=False)
    if not accepted_clusters:
        status = "NO_GUESS"
    elif unique_match:
        status = "OK"
    elif len(accepted_clusters) == 1:
        status = "WRONG_CLUSTER"
    else:
        status = "MULTIPLE_CLUSTERS"
    message = f"{len(rows)} accepted fit(s) from six localization clusters"
    return status, message, accepted_clusters, results_file


def write_summary(summary, summary_file: Path):
    rows = list(summary.values())
    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(summary_file, index=False)


def main(argv=None):
    args = parse_args(argv)
    simulation_model = args.simulation_model
    localization_models = [
        model_name for model_name in MODEL_ORDER if model_name != simulation_model
    ]
    if args.cluster is not None:
        source = read_cluster_source(args.csv, args.cluster)
        selection_name = f"cluster{args.cluster}"
    else:
        source = read_source(args.csv, args.row)
        selection_name = f"row{args.row}"

    root_dir = (
        Path(os.environ.get("OUT_DIR", ".")).resolve()
        / f"mismatched_{simulation_model}_{selection_name}"
    )
    event_dir = root_dir / "injected_events"
    event_dir.mkdir(parents=True, exist_ok=True)
    summary_file = root_dir / "mismatched_pair_summary.csv"
    summary = {}
    for localization_model in localization_models:
        pair_dir = root_dir / pair_prefix(simulation_model, localization_model)
        summary[localization_model] = {
            "injection_model": simulation_model,
            "localization_model": localization_model,
            "source_csv": str(source.csv_path),
            "source_row": source.row_index,
            "injected_cluster": source.cluster_index,
            "status": "PREPARING" if args.prepare_only else "STARTED",
            "message": "loading maps and injecting event",
            "accepted_clusters": "[]",
            "event_file": "",
            "results_file": "",
            "out_dir": str(pair_dir),
        }
        if not args.prepare_only:
            update_source(
                source,
                simulation_model,
                localization_model,
                run_status="STARTED",
                run_msg="analysis in progress",
                accepted_clusters="[]",
                event_file="",
                results_file="",
                out_dir=str(pair_dir),
            )
    write_summary(summary, summary_file)

    try:
        event = simulate_event(simulation_model, source, event_dir)
    except Exception as exc:
        message = f"injection failed: {type(exc).__name__}: {exc}"
        for localization_model in localization_models:
            summary[localization_model].update(status="ERROR", message=message)
            if not args.prepare_only:
                update_source(
                    source,
                    simulation_model,
                    localization_model,
                    run_status="ERROR",
                    run_msg=message,
                )
        write_summary(summary, summary_file)
        raise RuntimeError(message) from exc

    image_count = len(event.dt_true)
    print(f"Injected event contains {image_count} images.", flush=True)
    for record in summary.values():
        record["event_file"] = str(event.event_file)

    for localization_model in localization_models:
        localization_data = None
        try:
            localization_data = build_lensing_data(
                localization_model, SIGMA_DT_FALLBACK, MAX_MAP_PIXELS
            )
            if args.prepare_only:
                localization_system = ClusterLensing(
                    data=localization_data,
                    z_s_ref=source.params["z_s"],
                    cosmo_H0=source.params["H0"],
                )
                del localization_system
                message = (
                    f"prepared {image_count}-image injection and six-cluster "
                    f"{localization_model} recovery model"
                )
                summary[localization_model].update(
                    status="PREPARED",
                    message=message,
                )
                print(message, flush=True)
            else:
                pair_dir = Path(summary[localization_model]["out_dir"])
                status, message, accepted, results_file = run_pair(
                    simulation_model,
                    localization_model,
                    event,
                    localization_data,
                    pair_dir,
                )
                summary[localization_model].update(
                    status=status,
                    message=message,
                    accepted_clusters=json.dumps(accepted),
                    results_file=str(results_file),
                )
                update_source(
                    source,
                    simulation_model,
                    localization_model,
                    run_status=status,
                    run_msg=message,
                    accepted_clusters=json.dumps(accepted),
                    event_file=str(event.event_file),
                    results_file=str(results_file),
                )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            summary[localization_model].update(
                status="ERROR",
                message=message,
                event_file=str(event.event_file),
            )
            if not args.prepare_only:
                update_source(
                    source,
                    simulation_model,
                    localization_model,
                    run_status="ERROR",
                    run_msg=message,
                    event_file=str(event.event_file),
                )
            print(f"{simulation_model} -> {localization_model} failed: {message}", flush=True)
        finally:
            if localization_data is not None:
                del localization_data
            gc.collect()
            write_summary(summary, summary_file)

    failures = [record for record in summary.values() if record["status"] == "ERROR"]
    print(f"\nSummary written to {summary_file}", flush=True)
    if failures:
        failed_pairs = ", ".join(
            pair_prefix(record["injection_model"], record["localization_model"])
            for record in failures
        )
        raise RuntimeError(f"failed mismatched pairs: {failed_pairs}")


if __name__ == "__main__":
    main()
