"""Generate four-image CATS, Keeton, and Sharon source catalogs."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_local_tidy import ClusterLensing
from simulation_diff import CLUSTERS, build_lensing_data


SCRIPT_DIR = Path(__file__).resolve().parent
MODELS = ("sharon", "cats", "keeton")
OUTPUT_NAMES = {
    model: f"src_pos_tidy_xyz_{model}.csv" for model in MODELS
}

SAMPLES_PER_CLUSTER = 1
TARGET_IMAGE_COUNT = 3
SOURCE_REDSHIFT_BOUNDS = (1.5, 4.0)
H0 = 70.0
SEARCH_RANGES_ARCSEC = (30.0, 50.0, 50.0, 50.0, 100.0, 60.0)
SIGMA_DT_FALLBACK = 0.1
MAX_MAP_PIXELS = 1600
MAX_ATTEMPTS_PER_CLUSTER = 20000
CSV_COLUMNS = [
    "indices",
    "x",
    "y",
    "z",
    "localized_index",
    "localized_x",
    "localized_y",
    "localized_z",
    "run_status",
    "run_msg",
    "localized_H0",
    "chi_sq",
    "accepted_clusters",
    "posterior_file",
    "corner_plot",
    "trace_plot",
    "second_smallest_chi2",
    "bayes_factor",
    "posterior_odds",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="directory for the three CSV files",
    )
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace generated catalogs that already exist",
    )
    return parser.parse_args(argv)


def empty_record():
    return {column: None for column in CSV_COLUMNS}


def generate_cluster_rows(system, cluster_index, rng):
    pixscale = float(system.data.pixscale[cluster_index])
    x_center = float(system.data.x_center[cluster_index])
    y_center = float(system.data.y_center[cluster_index])
    ny, nx = system.data.alpha_maps_x[cluster_index].shape
    width = (nx - 1) * pixscale
    height = (ny - 1) * pixscale
    search_range = SEARCH_RANGES_ARCSEC[cluster_index]

    x_bounds = (
        max(pixscale, x_center - search_range),
        min(width - pixscale, x_center + search_range),
    )
    y_bounds = (
        max(pixscale, y_center - search_range),
        min(height - pixscale, y_center + search_range),
    )
    if x_bounds[0] >= x_bounds[1] or y_bounds[0] >= y_bounds[1]:
        raise ValueError(f"empty source search region for cluster {cluster_index}")

    rows = []
    for attempt in range(1, MAX_ATTEMPTS_PER_CLUSTER + 1):
        x_src = rng.uniform(*x_bounds)
        y_src = rng.uniform(*y_bounds)
        z_src = rng.uniform(*SOURCE_REDSHIFT_BOUNDS)
        params = {"x_src": x_src, "y_src": y_src, "z_s": z_src, "H0": H0}
        event = system.calculate_imgs_delays_magns(params, cluster_index)
        if len(event["image_positions"][0]) != TARGET_IMAGE_COUNT:
            continue

        record = empty_record()
        record.update(
            indices=cluster_index,
            x=float(x_src),
            y=float(y_src),
            z=float(z_src),
        )
        rows.append(record)
        print(
            f"  cluster {cluster_index}: accepted {len(rows)}/{SAMPLES_PER_CLUSTER} "
            f"after {attempt} attempts",
            flush=True,
        )
        if len(rows) == SAMPLES_PER_CLUSTER:
            system.clear_candidate_resources(cluster_index)
            return rows

    raise RuntimeError(
        f"cluster {cluster_index}: found only {len(rows)} valid sources after "
        f"{MAX_ATTEMPTS_PER_CLUSTER} attempts"
    )


def generate_model_catalog(model_name, rng, output_path):
    print(f"\nLoading {model_name.upper()} maps", flush=True)
    data = build_lensing_data(
        model_name,
        sigma_fallback=SIGMA_DT_FALLBACK,
        max_map_pixels=MAX_MAP_PIXELS,
    )
    system = ClusterLensing(data=data, z_s_ref=1.5, cosmo_H0=H0)

    rows = []
    for cluster_index, cluster in enumerate(CLUSTERS):
        print(
            f"{model_name.upper()} {cluster.key}: "
            f"pixscale={data.pixscale[cluster_index]:.8g} arcsec/pixel, "
            f"center=({data.x_center[cluster_index]:.5g}, "
            f"{data.y_center[cluster_index]:.5g}) arcsec",
            flush=True,
        )
        rows.extend(generate_cluster_rows(system, cluster_index, rng))

    frame = pd.DataFrame(rows, columns=CSV_COLUMNS)
    frame.to_csv(output_path, index=False)
    print(f"Saved {len(frame)} rows to {output_path}", flush=True)
    del system, data
    gc.collect()


def validate_catalog(path):
    frame = pd.read_csv(path)
    if list(frame.columns) != CSV_COLUMNS:
        raise RuntimeError(f"unexpected columns in {path}")
    expected_indices = np.repeat(np.arange(len(CLUSTERS)), SAMPLES_PER_CLUSTER)
    if len(frame) != len(expected_indices):
        raise RuntimeError(f"unexpected row count in {path}: {len(frame)}")
    if not np.array_equal(frame["indices"].to_numpy(), expected_indices):
        raise RuntimeError(f"unexpected cluster ordering in {path}")
    if frame[["x", "y", "z"]].isna().any().any():
        raise RuntimeError(f"missing source parameters in {path}")


def main(argv=None):
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        model: output_dir / OUTPUT_NAMES[model] for model in MODELS
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output already exists: {names}; use --overwrite")

    seed_sequences = np.random.SeedSequence(args.seed).spawn(len(MODELS))
    for model_name, seed_sequence in zip(MODELS, seed_sequences):
        generate_model_catalog(
            model_name,
            np.random.default_rng(seed_sequence),
            output_paths[model_name],
        )
        validate_catalog(output_paths[model_name])

    print("\nGenerated catalogs:", flush=True)
    for path in output_paths.values():
        print(f"  {path}", flush=True)


if __name__ == "__main__":
    main()
