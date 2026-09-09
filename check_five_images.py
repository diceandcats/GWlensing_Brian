"""CATS-to-Keeton five-image check."""
import time

import numpy as np
import pandas as pd

from cluster_local_tidy import ClusterLensing
from lensing_data_class import LensingData
from simulation_diff import CLUSTERS, load_cluster_maps

ROW, RADIUS, STEP, H0 = 25, 1.50, 0.05, 70.0
PROGRESS_EVERY = 50
RUN_STARTED = time.monotonic()


def report_progress(message):
    """Print a log-friendly progress message with total elapsed time."""
    elapsed = int(time.monotonic() - RUN_STARTED)
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(
        f"[progress +{hours:02d}:{minutes:02d}:{seconds:02d}] {message}",
        flush=True,
    )


report_progress(f"Reading catalog row {ROW}")
row = pd.read_csv("src_pos_tidy_xyz.csv").iloc[ROW]
cluster, z = CLUSTERS[int(row["indices"])], float(row["z"])


def model(name):
    ax, ay, psi, sigma, pix, xc, yc = load_cluster_maps(cluster, name, 0.1, 1600)
    data = LensingData([ax], [ay], [psi], [sigma], [pix],
                       [cluster.lens_redshift], [xc], [yc], [cluster.search_window])
    lens = ClusterLensing(data, z, cosmo_H0=H0)
    kwargs = lens._get_scaled_kwargs(z, 0, H0)
    lens_model, solver = lens._get_lens_and_solver(0, z, H0)
    alpha = lens_model.alpha(xc, yc, [kwargs])
    beta_ref = np.array([xc - alpha[0], yc - alpha[1]])

    def image_count(x, y):
        return len(solver.image_position_from_source(
            x, y, [kwargs], min_distance=pix, search_window=cluster.search_window,
            x_center=xc, y_center=yc)[0])

    return beta_ref, image_count


report_progress(f"Loading CATS maps for {cluster.key} (model 1/2)")
cats_ref, cats_count = model("cats")
report_progress(f"Loading Keeton maps for {cluster.key} (model 2/2)")
keeton_ref, keeton_count = model("keeton")
cats_position = np.array([row["x"], row["y"]], dtype=float)
# Match source offsets from beta(theta_center), removing each model's gauge shift.
keeton_position = keeton_ref + cats_position - cats_ref
report_progress("Checking the CATS source and gauge-matched Keeton source")
assert cats_count(*cats_position) == 5, "the hard-coded CATS source is not five-imaged"
exact_keeton_count = keeton_count(*keeton_position)

offset = np.arange(-RADIUS, RADIUS + STEP / 2, STEP)
points = sorted((np.hypot(dx, dy), *(keeton_position + (dx, dy)))
                for dx in offset for dy in offset if np.hypot(dx, dy) <= RADIUS)
report_progress(
    f"Searching {len(points)} Keeton positions within {RADIUS:g} arcsec "
    f"at {STEP:g}-arcsec spacing"
)
match = None
search_started = time.monotonic()
for tested, (distance, x, y) in enumerate(points, start=1):
    if keeton_count(x, y) == 5:
        match = (distance, x, y)
        report_progress(
            f"Found a five-image position after {tested}/{len(points)} tests "
            f"({100 * tested / len(points):.1f}%)"
        )
        break
    if tested % PROGRESS_EVERY == 0 or tested == len(points):
        search_elapsed = time.monotonic() - search_started
        rate = tested / search_elapsed if search_elapsed > 0 else float("inf")
        report_progress(
            f"Grid search: tested {tested}/{len(points)} positions "
            f"({100 * tested / len(points):.1f}%, {rate:.2f} positions/s)"
        )

if match is None:
    report_progress("Grid search complete; no five-image Keeton position was found")

print(f"CATS row {ROW}: 5 images at {cats_position}")
print(f"Corresponding Keeton position: {keeton_position} ({exact_keeton_count} images)")
print(f"Nearest Keeton 5-image position within {RADIUS} arcsec on the {STEP}-arcsec grid: {match}")
report_progress("Run complete")
