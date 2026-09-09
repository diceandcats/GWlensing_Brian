#!/usr/bin/env python3
"""Validate whether diego range kappa maps reproduce *ensemble variations* in gamma.

This script replaces the earlier per-realization constant-external-shear fit as the
primary range-map test.  The earlier test asked whether every finite kappa cutout
could reproduce the *absolute* diego scalar-gamma map after adding one constant
shear vector.  That is not the quantity needed for an uncertainty ensemble.

The test here is instead:

    delta kappa_i = kappa_i - <kappa>_range

    delta gamma_vec_i = gamma_vec[kappa_i] - gamma_vec[<kappa>_range]

and the predicted shear realization is anchored to the independent best-fit
diego vector-shear field:

    gamma1_pred_i = gamma1_best + delta gamma1_i
    gamma2_pred_i = gamma2_best + delta gamma2_i
    |gamma|_pred_i = hypot(gamma1_pred_i, gamma2_pred_i)

The predicted scalar |gamma| ensemble is then compared with the actual paired
diego range gamma maps in the quantities that matter for an uncertainty model:

  * global realization-to-realization fluctuation correlation;
  * global fluctuation relative RMS;
  * per-pixel sigma_gamma map agreement;
  * per-pixel correlation of realization fluctuations;
  * absolute |gamma| agreement, retained only as a secondary diagnostic.

Why this test is more relevant
------------------------------
A spatial field common to every realization can make the absolute finite-field
kappa->gamma reconstruction look poor while cancelling from the ensemble
scatter.  sigma_dt ultimately depends on the model-to-model variation, so the
range validation should primarily test whether delta-kappa predicts delta-gamma,
not whether every realization independently reproduces the absolute shear field.

Best-fit potential validation is retained
-----------------------------------------
The script also reconstructs the best-fit psi from the best-fit kappa map, fits
the low-order harmonic finite-field correction from the official deflection
maps, and checks potential differences against line integrals of the official
alpha field.  This establishes the anchor potential used if candidate range psi
maps are written.

Candidate range psi maps
------------------------
Only with --save-range-psi, candidate potentials are written as

    psi_candidate_i = psi_best,harmonic(on range grid)
                      + [psi(kappa_i) - psi(<kappa>_range)].

This deliberately does NOT fit a scalar-gamma correction into each potential.
The paired scalar-gamma maps are used to validate the ensemble variation; scalar
|gamma| alone is not treated as a unique source of the shear-vector orientation.

This script does not calculate sigma_dt and does not modify simulation_diff.py.
A passing result is evidence to proceed to a separate gauge-consistent
Fermat-potential/sigma_dt validation, not an automatic production approval.

Requirements
------------
    numpy scipy astropy matplotlib lenstronomy

Example
-------
    python validate_diego_ensemble_variation.py \
        --diego-dir "GCdata/Abell 2744/diego" \
        --output-dir "diego_ensemble_validation/abell2744"

After inspecting a passing validation, optionally write candidate range psi maps:

    python validate_diego_ensemble_variation.py \
        --diego-dir "GCdata/Abell 2744/diego" \
        --output-dir "diego_ensemble_validation/abell2744" \
        --save-range-psi
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import map_coordinates

from lenstronomy.LensModel.convergence_integrals import (
    deflection_from_kappa_grid,
    potential_from_kappa_grid,
)


MAP_ID_RE = re.compile(r"diego-map(\d+)", re.IGNORECASE)


@dataclass
class Reconstruction:
    psi: np.ndarray | None
    alpha_x: np.ndarray
    alpha_y: np.ndarray
    kappa: np.ndarray
    gamma1: np.ndarray
    gamma2: np.ndarray


@dataclass
class HarmonicCorrection:
    bx: float
    by: float
    kappa0: float
    gamma1: float
    gamma2: float

    def potential(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (
            self.bx * x
            + self.by * y
            + 0.5 * self.kappa0 * (x**2 + y**2)
            + 0.5 * self.gamma1 * (x**2 - y**2)
            + self.gamma2 * x * y
        )

    def alpha(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ax = self.bx + (self.kappa0 + self.gamma1) * x + self.gamma2 * y
        ay = self.by + self.gamma2 * x + (self.kappa0 - self.gamma1) * y
        return ax, ay


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diego-dir", type=Path, required=True)
    p.add_argument("--range-dir", type=Path, default=None)
    p.add_argument(
        "--output-dir", type=Path, default=Path("diego_ensemble_variation_validation")
    )
    p.add_argument(
        "--edge-fraction", type=float, default=0.15,
        help="Trim this fraction from each edge for quantitative tests (default 0.15).",
    )
    p.add_argument(
        "--pad-factor", type=float, default=1.0,
        help="Extra zero-padding before lenstronomy convolution (default 1).",
    )
    p.add_argument(
        "--fit-stride", type=int, default=4,
        help="Stride for best-fit low-order alpha-residual fit (default 4).",
    )
    p.add_argument("--pair-tests", type=int, default=500)
    p.add_argument("--line-samples", type=int, default=128)
    p.add_argument("--max-range", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260810)
    p.add_argument(
        "--save-range-psi", action="store_true",
        help="Write candidate range psi maps only after/alongside validation.",
    )
    p.add_argument(
        "--wcs-chunk-rows", type=int, default=128,
        help="Rows per chunk when resampling best-fit maps to range WCS.",
    )

    # These are screening thresholds, not physical theorems.
    p.add_argument("--decision-bestfit-dpsi-rms", type=float, default=0.02)
    p.add_argument("--decision-fluct-rms", type=float, default=0.25)
    p.add_argument("--decision-fluct-corr", type=float, default=0.90)
    p.add_argument("--decision-sigma-rms", type=float, default=0.25)
    p.add_argument("--decision-sigma-corr", type=float, default=0.90)

    args = p.parse_args()
    if not 0 <= args.edge_fraction < 0.45:
        p.error("--edge-fraction must be in [0, 0.45)")
    if args.pad_factor < 1.0:
        p.error("--pad-factor must be >= 1")
    if args.fit_stride < 1:
        p.error("--fit-stride must be >= 1")
    if args.wcs_chunk_rows < 1:
        p.error("--wcs-chunk-rows must be >= 1")
    return args


def read_map(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header.copy()
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D FITS map: {path}; got {data.shape}")
    return data, header


def pixel_scale_arcsec(header: fits.Header) -> float:
    w = WCS(header).celestial
    scales = np.asarray(proj_plane_pixel_scales(w), dtype=float) * 3600.0
    if scales.size != 2 or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Could not determine a valid celestial pixel scale")
    if not np.isclose(scales[0], scales[1], rtol=5e-3):
        raise ValueError(f"Anisotropic pixels not supported: {scales}")
    return float(np.mean(scales))


def wcs_orientation_deg(header: fits.Header) -> float:
    """Approximate orientation of the image +x axis in the projected sky plane."""
    matrix = np.asarray(WCS(header).celestial.pixel_scale_matrix, dtype=float)
    if matrix.shape != (2, 2) or np.any(~np.isfinite(matrix)):
        raise ValueError("Could not determine celestial WCS orientation")
    return float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))


def angle_difference_deg(a: float, b: float) -> float:
    """Smallest signed difference a-b in degrees, modulo 360."""
    return float((a - b + 180.0) % 360.0 - 180.0)


def find_unique(directory: Path, suffix: str, required: bool = True) -> Path | None:
    # Avoid accidentally matching range files if a recursive glob is ever used.
    matches = [
        p for p in directory.glob(f"*_{suffix}.fits")
        if "diego-map" not in p.name.lower()
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


def discover_range_pairs(range_dir: Path) -> list[tuple[str, Path, Path]]:
    kappas: dict[str, Path] = {}
    gammas: dict[str, Path] = {}
    for path in range_dir.glob("*.fits"):
        match = MAP_ID_RE.search(path.name)
        if not match:
            continue
        map_id = match.group(1)
        lower = path.name.lower()
        if lower.endswith("_kappa.fits"):
            kappas[map_id] = path
        elif lower.endswith("_gamma.fits") and not lower.endswith("_gamma1.fits") and not lower.endswith("_gamma2.fits"):
            gammas[map_id] = path
    common = sorted(set(kappas) & set(gammas), key=lambda s: int(s))
    if not common:
        raise FileNotFoundError(f"No paired diego-mapXXX kappa/gamma FITS files in {range_dir}")
    missing_g = sorted(set(kappas) - set(gammas), key=lambda s: int(s))
    missing_k = sorted(set(gammas) - set(kappas), key=lambda s: int(s))
    if missing_g or missing_k:
        raise RuntimeError(
            "Unpaired range maps found: "
            f"missing gamma for {missing_g}; missing kappa for {missing_k}"
        )
    return [(i, kappas[i], gammas[i]) for i in common]


def edge_mask(shape: tuple[int, int], fraction: float) -> np.ndarray:
    ny, nx = shape
    ey = int(np.floor(ny * fraction))
    ex = int(np.floor(nx * fraction))
    mask = np.zeros(shape, dtype=bool)
    mask[ey:ny-ey if ey else ny, ex:nx-ex if ex else nx] = True
    return mask


def coordinate_grids(shape: tuple[int, int], pixscale: float) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    x = (np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)) * pixscale
    y = (np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)) * pixscale
    return np.meshgrid(x, y)


def pad_widths(shape: tuple[int, int], factor: float) -> tuple[tuple[int, int], tuple[int, int]]:
    if factor <= 1.0:
        return ((0, 0), (0, 0))
    target = [int(np.ceil(n * factor)) for n in shape]
    pads = []
    for n, t in zip(shape, target):
        extra = max(0, t - n)
        before = extra // 2
        pads.append((before, extra - before))
    return pads[0], pads[1]


def crop_from_padding(data: np.ndarray, pads: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    (y0, y1), (x0, x1) = pads
    ys = slice(y0, data.shape[0] - y1 if y1 else None)
    xs = slice(x0, data.shape[1] - x1 if x1 else None)
    return np.asarray(data[ys, xs], dtype=np.float64)


def reconstruct_from_kappa(
    kappa: np.ndarray,
    pixscale: float,
    pad_factor: float,
    need_potential: bool,
) -> Reconstruction:
    """Reconstruct alpha/shear, and optionally psi, from a kappa map."""
    pads = pad_widths(kappa.shape, pad_factor)
    work = np.pad(kappa, pads, mode="constant") if pads != ((0, 0), (0, 0)) else kappa

    ax_pad, ay_pad = deflection_from_kappa_grid(work, pixscale)
    alpha_x = crop_from_padding(np.asarray(ax_pad, dtype=np.float64), pads)
    alpha_y = crop_from_padding(np.asarray(ay_pad, dtype=np.float64), pads)

    d_ax_dy, d_ax_dx = np.gradient(alpha_x, pixscale, pixscale)
    d_ay_dy, d_ay_dx = np.gradient(alpha_y, pixscale, pixscale)
    kappa_rec = 0.5 * (d_ax_dx + d_ay_dy)
    gamma1 = 0.5 * (d_ax_dx - d_ay_dy)
    gamma2 = 0.5 * (d_ax_dy + d_ay_dx)

    psi = None
    if need_potential:
        psi_pad = potential_from_kappa_grid(work, pixscale)
        psi = crop_from_padding(np.asarray(psi_pad, dtype=np.float64), pads)

    return Reconstruction(
        psi=psi,
        alpha_x=alpha_x,
        alpha_y=alpha_y,
        kappa=kappa_rec,
        gamma1=gamma1,
        gamma2=gamma2,
    )


def scalar_relative_rms(diff: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(diff) & np.isfinite(reference)
    num = float(np.sqrt(np.mean(diff[valid] ** 2)))
    den = float(np.sqrt(np.mean(reference[valid] ** 2)))
    return num / den if den > 0 else np.nan


def vector_relative_rms(
    dx: np.ndarray, dy: np.ndarray,
    ref_x: np.ndarray, ref_y: np.ndarray,
    mask: np.ndarray,
) -> float:
    valid = mask & np.isfinite(dx) & np.isfinite(dy) & np.isfinite(ref_x) & np.isfinite(ref_y)
    num = float(np.sqrt(np.mean(dx[valid] ** 2 + dy[valid] ** 2)))
    den = float(np.sqrt(np.mean(ref_x[valid] ** 2 + ref_y[valid] ** 2)))
    return num / den if den > 0 else np.nan


def fit_potential_correction(
    residual_x: np.ndarray,
    residual_y: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    stride: int,
    allow_kappa0: bool,
) -> HarmonicCorrection:
    sample = np.zeros_like(mask)
    sample[::stride, ::stride] = True
    sample &= mask
    valid = sample & np.isfinite(residual_x) & np.isfinite(residual_y)

    xx = x[valid]
    yy = y[valid]
    rx = residual_x[valid]
    ry = residual_y[valid]

    # ax = bx + (k0+g1)x + g2 y
    # ay = by + g2 x + (k0-g1)y
    if allow_kappa0:
        # columns: bx, by, k0, g1, g2
        ax_design = np.column_stack((np.ones_like(xx), np.zeros_like(xx), xx, xx, yy))
        ay_design = np.column_stack((np.zeros_like(xx), np.ones_like(xx), yy, -yy, xx))
    else:
        # columns: bx, by, g1, g2
        ax_design = np.column_stack((np.ones_like(xx), np.zeros_like(xx), xx, yy))
        ay_design = np.column_stack((np.zeros_like(xx), np.ones_like(xx), -yy, xx))

    design = np.vstack((ax_design, ay_design))
    target = np.concatenate((rx, ry))
    coeff, *_ = np.linalg.lstsq(design, target, rcond=None)
    if allow_kappa0:
        bx, by, k0, g1, g2 = coeff
    else:
        bx, by, g1, g2 = coeff
        k0 = 0.0
    return HarmonicCorrection(float(bx), float(by), float(k0), float(g1), float(g2))


def apply_correction(
    rec: Reconstruction,
    correction: HarmonicCorrection,
    x: np.ndarray,
    y: np.ndarray,
) -> Reconstruction:
    dax, day = correction.alpha(x, y)
    psi = None if rec.psi is None else rec.psi + correction.potential(x, y)
    return Reconstruction(
        psi=psi,
        alpha_x=rec.alpha_x + dax,
        alpha_y=rec.alpha_y + day,
        kappa=rec.kappa + correction.kappa0,
        gamma1=rec.gamma1 + correction.gamma1,
        gamma2=rec.gamma2 + correction.gamma2,
    )


def sample_map(data: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    return map_coordinates(data, [row, col], order=1, mode="nearest", prefilter=False)


def line_integral_potential_tests(
    alpha_x: np.ndarray,
    alpha_y: np.ndarray,
    psi_maps: dict[str, np.ndarray],
    mask: np.ndarray,
    pixscale: float,
    n_pairs: int,
    n_samples: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    rows, cols = np.nonzero(mask)
    if rows.size < 2:
        return {}
    n_pairs = min(n_pairs, rows.size // 2 if rows.size > 2 else 1)
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

    out: dict[str, dict[str, float]] = {}
    for name, psi in psi_maps.items():
        psi_a = sample_map(psi, ra, ca)
        psi_b = sample_map(psi, rb, cb)
        dpsi = psi_b - psi_a
        residual = dpsi - line
        denom = float(np.sqrt(np.mean(line**2)))
        rel = float(np.sqrt(np.mean(residual**2)) / denom) if denom > 0 else np.nan
        corr = float(np.corrcoef(dpsi, line)[0, 1]) if n_pairs > 1 else np.nan
        out[name] = {
            "relative_rms": rel,
            "median_abs_arcsec2": float(np.median(np.abs(residual))),
            "p95_abs_arcsec2": float(np.percentile(np.abs(residual), 95)),
            "correlation": corr,
        }
    return out


def resample_wcs(
    source: np.ndarray,
    source_header: fits.Header,
    target_header: fits.Header,
    target_shape: tuple[int, int],
    chunk_rows: int,
) -> np.ndarray:
    """Bilinearly sample a source celestial FITS map onto a target celestial WCS."""
    sw = WCS(source_header).celestial
    tw = WCS(target_header).celestial
    if not sw.has_celestial or not tw.has_celestial:
        raise ValueError("Both source and target FITS headers need celestial WCS")

    ny, nx = target_shape
    out = np.empty(target_shape, dtype=np.float64)
    xpix = np.arange(nx, dtype=np.float64)
    for y0 in range(0, ny, chunk_rows):
        y1 = min(ny, y0 + chunk_rows)
        yy, xx = np.meshgrid(np.arange(y0, y1, dtype=np.float64), xpix, indexing="ij")
        lon, lat = tw.pixel_to_world_values(xx, yy)
        sx, sy = sw.world_to_pixel_values(lon, lat)
        vals = map_coordinates(
            source,
            [sy.ravel(), sx.ravel()],
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )
        out[y0:y1] = vals.reshape(y1 - y0, nx)
    return out


def correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(a) & np.isfinite(b)
    aa = a[valid]
    bb = b[valid]
    if aa.size < 2 or np.std(aa) == 0 or np.std(bb) == 0:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_bestfit_plot(
    path: Path,
    kappa: np.ndarray,
    gamma_ref: np.ndarray | None,
    alpha_x_ref: np.ndarray,
    alpha_y_ref: np.ndarray,
    rec_raw: Reconstruction,
    rec_h: Reconstruction,
    mask: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fields = [
        (kappa, "diego best-fit kappa"),
        (gamma_ref if gamma_ref is not None else np.hypot(rec_raw.gamma1, rec_raw.gamma2), "diego best-fit |gamma|"),
        (np.hypot(alpha_x_ref, alpha_y_ref), "diego |alpha|"),
        (np.hypot(rec_raw.alpha_x-alpha_x_ref, rec_raw.alpha_y-alpha_y_ref), "|alpha_rec-alpha| raw"),
        (np.hypot(rec_h.alpha_x-alpha_x_ref, rec_h.alpha_y-alpha_y_ref), "|alpha_rec-alpha| harmonic"),
        (mask.astype(float), "comparison mask"),
    ]
    for ax, (data, title) in zip(axes.ravel(), fields):
        im = ax.imshow(data, origin="lower")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_ensemble_plot(
    path: Path,
    mean_obs: np.ndarray,
    mean_pred: np.ndarray,
    sigma_obs: np.ndarray,
    sigma_pred: np.ndarray,
    corr_map: np.ndarray,
    mask: np.ndarray,
) -> None:
    valid_sigma = mask & np.isfinite(sigma_obs) & np.isfinite(sigma_pred)
    sigma_res = np.full_like(sigma_obs, np.nan)
    floor = np.nanpercentile(sigma_obs[valid_sigma], 10) if np.any(valid_sigma) else 0.0
    denom = np.maximum(sigma_obs, floor if floor > 0 else 1e-12)
    sigma_res[valid_sigma] = (sigma_pred[valid_sigma] - sigma_obs[valid_sigma]) / denom[valid_sigma]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    items = [
        (mean_obs, "Observed range mean |gamma|"),
        (mean_pred, "Predicted range mean |gamma|"),
        (sigma_obs, "Observed sigma_gamma"),
        (sigma_pred, "Predicted sigma_gamma"),
        (sigma_res, "(sigma_pred-sigma_obs)/sigma_obs"),
        (corr_map, "Per-pixel fluctuation correlation"),
    ]
    for ax, (data, title) in zip(axes.ravel(), items):
        im = ax.imshow(data, origin="lower")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_sigma_scatter(
    path: Path,
    sigma_obs: np.ndarray,
    sigma_pred: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    max_points: int = 50000,
) -> None:
    valid = mask & np.isfinite(sigma_obs) & np.isfinite(sigma_pred) & (sigma_obs >= 0) & (sigma_pred >= 0)
    x = sigma_obs[valid]
    y = sigma_pred[valid]
    if x.size > max_points:
        idx = rng.choice(x.size, size=max_points, replace=False)
        x, y = x[idx], y[idx]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, s=3, alpha=0.2)
    hi = float(max(np.max(x), np.max(y))) if x.size else 1.0
    ax.plot([0, hi], [0, hi], linestyle="--")
    ax.set_xlabel("Observed sigma_gamma")
    ax.set_ylabel("Predicted sigma_gamma")
    ax.set_title("Per-pixel ensemble uncertainty")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    diego_dir = args.diego_dir.resolve()
    range_dir = (args.range_dir or (diego_dir / "range")).resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    kappa_path = find_unique(diego_dir, "kappa")
    gamma_path = find_unique(diego_dir, "gamma", required=False)
    gamma1_path = find_unique(diego_dir, "gamma1", required=False)
    gamma2_path = find_unique(diego_dir, "gamma2", required=False)
    alpha_x_path = find_unique(diego_dir, "x-arcsec-deflect")
    alpha_y_path = find_unique(diego_dir, "y-arcsec-deflect")

    kappa_ref, ref_header = read_map(kappa_path)
    alpha_x_ref, ax_header = read_map(alpha_x_path)
    alpha_y_ref, ay_header = read_map(alpha_y_path)
    gamma_ref, gamma_header = read_map(gamma_path) if gamma_path else (None, None)
    gamma1_ref, gamma1_header = read_map(gamma1_path) if gamma1_path else (None, None)
    gamma2_ref, gamma2_header = read_map(gamma2_path) if gamma2_path else (None, None)

    for arr in [alpha_x_ref, alpha_y_ref] + ([gamma_ref] if gamma_ref is not None else []) + ([gamma1_ref] if gamma1_ref is not None else []) + ([gamma2_ref] if gamma2_ref is not None else []):
        if arr.shape != kappa_ref.shape:
            raise ValueError(f"Best-fit map shape mismatch: kappa={kappa_ref.shape}, other={arr.shape}")

    pix_ref = pixel_scale_arcsec(ref_header)
    mask_ref = edge_mask(kappa_ref.shape, args.edge_fraction)
    x_ref, y_ref = coordinate_grids(kappa_ref.shape, pix_ref)

    print(f"Best-fit kappa: {kappa_path.name}")
    print(f"Shape: {kappa_ref.shape}")
    print(f"Pixel scale: {pix_ref:.8g} arcsec/pixel")
    print(f"FFT zero-padding factor: {args.pad_factor:g}")

    rec_ref = reconstruct_from_kappa(kappa_ref, pix_ref, args.pad_factor, need_potential=True)
    if rec_ref.psi is None:
        raise RuntimeError("Internal error: best-fit psi was not reconstructed")

    residual_x = alpha_x_ref - rec_ref.alpha_x
    residual_y = alpha_y_ref - rec_ref.alpha_y
    harmonic = fit_potential_correction(
        residual_x, residual_y, x_ref, y_ref, mask_ref, args.fit_stride, allow_kappa0=False
    )
    rec_ref_h = apply_correction(rec_ref, harmonic, x_ref, y_ref)
    full_quad = fit_potential_correction(
        residual_x, residual_y, x_ref, y_ref, mask_ref, args.fit_stride, allow_kappa0=True
    )
    rec_ref_full = apply_correction(rec_ref, full_quad, x_ref, y_ref)

    alpha_raw_rms = vector_relative_rms(
        rec_ref.alpha_x-alpha_x_ref, rec_ref.alpha_y-alpha_y_ref,
        alpha_x_ref, alpha_y_ref, mask_ref,
    )
    alpha_h_rms = vector_relative_rms(
        rec_ref_h.alpha_x-alpha_x_ref, rec_ref_h.alpha_y-alpha_y_ref,
        alpha_x_ref, alpha_y_ref, mask_ref,
    )
    kappa_closure = scalar_relative_rms(rec_ref.kappa-kappa_ref, kappa_ref, mask_ref)

    line_tests = line_integral_potential_tests(
        alpha_x_ref,
        alpha_y_ref,
        {"raw": rec_ref.psi, "harmonic": rec_ref_h.psi, "full_quad": rec_ref_full.psi},
        mask_ref,
        pix_ref,
        args.pair_tests,
        args.line_samples,
        rng,
    )
    bestfit_dpsi_rms = line_tests.get("harmonic", {}).get("relative_rms", np.nan)

    if gamma1_ref is None or gamma2_ref is None:
        # Derive an independent vector-shear anchor from the official alpha field.
        d_ax_dy, d_ax_dx = np.gradient(alpha_x_ref, pix_ref, pix_ref)
        d_ay_dy, d_ay_dx = np.gradient(alpha_y_ref, pix_ref, pix_ref)
        gamma1_ref = 0.5 * (d_ax_dx - d_ay_dy)
        gamma2_ref = 0.5 * (d_ax_dy + d_ay_dx)
        gamma_anchor_source = "derived from official best-fit alpha"
    else:
        gamma_anchor_source = "official best-fit gamma1/gamma2 FITS"

    print("\n=== BEST-FIT ANCHOR VALIDATION ===")
    print(f"kappa closure relative RMS: {kappa_closure:.6g}")
    print(f"official alpha relative RMS raw:      {alpha_raw_rms:.6g}")
    print(f"official alpha relative RMS harmonic: {alpha_h_rms:.6g}")
    print(
        "harmonic correction: "
        f"bx={harmonic.bx:+.6g}, by={harmonic.by:+.6g}, "
        f"gamma1={harmonic.gamma1:+.6g}, gamma2={harmonic.gamma2:+.6g}"
    )
    for name, metrics in line_tests.items():
        print(
            f"potential-difference [{name}]: rel_rms={metrics['relative_rms']:.6g}, "
            f"corr={metrics['correlation']:.6g}"
        )

    save_bestfit_plot(
        out / "bestfit_validation.png",
        kappa_ref,
        gamma_ref,
        alpha_x_ref,
        alpha_y_ref,
        rec_ref,
        rec_ref_h,
        mask_ref,
    )

    # Save the validated anchor potential for inspection.
    anchor_header = ref_header.copy()
    anchor_header["BUNIT"] = "arcsec2"
    anchor_header.add_history("Best-fit psi reconstructed from kappa and harmonic alpha-residual correction")
    fits.PrimaryHDU(rec_ref_h.psi.astype(np.float32), header=anchor_header).writeto(
        out / "bestfit_psi_harmonic.fits", overwrite=True
    )

    if not range_dir.is_dir():
        raise FileNotFoundError(f"Range directory does not exist: {range_dir}")
    pairs = discover_range_pairs(range_dir)
    if args.max_range:
        pairs = pairs[:args.max_range]
    if len(pairs) < 2:
        raise ValueError("Need at least two range realizations for an ensemble-variation test")
    print(f"\nFound {len(pairs)} paired range kappa/gamma realizations in {range_dir}")

    # ------------------------------------------------------------------
    # Pass 1: establish the native range grid and compute <kappa>_range.
    # ------------------------------------------------------------------
    mean_kappa = None
    range_header = None
    range_gamma_header = None
    range_shape = None
    pix_range = None
    sum_gamma_obs_for_mean = None

    for j, (map_id, kp, gp) in enumerate(pairs, start=1):
        kappa_i, kh = read_map(kp)
        gamma_i, gh = read_map(gp)
        if kappa_i.shape != gamma_i.shape:
            raise ValueError(f"Range map {map_id} kappa/gamma shape mismatch: {kappa_i.shape} vs {gamma_i.shape}")
        pk = pixel_scale_arcsec(kh)
        pg = pixel_scale_arcsec(gh)
        if not np.isclose(pk, pg, rtol=5e-4):
            raise ValueError(f"Range map {map_id} kappa/gamma pixel-scale mismatch: {pk} vs {pg}")
        if range_shape is None:
            range_shape = kappa_i.shape
            pix_range = pk
            range_header = kh
            range_gamma_header = gh
            mean_kappa = np.zeros(range_shape, dtype=np.float64)
            sum_gamma_obs_for_mean = np.zeros(range_shape, dtype=np.float64)
            extent = ((range_shape[1]-1)*pix_range, (range_shape[0]-1)*pix_range)
            ref_extent = ((kappa_ref.shape[1]-1)*pix_ref, (kappa_ref.shape[0]-1)*pix_ref)
            print(
                f"Range grid: shape={range_shape}, pixscale={pix_range:.8g} arcsec/pixel, "
                f"extent={extent[0]:.6g} x {extent[1]:.6g} arcsec"
            )
            print(
                f"Best-fit grid: shape={kappa_ref.shape}, pixscale={pix_ref:.8g} arcsec/pixel, "
                f"extent={ref_extent[0]:.6g} x {ref_extent[1]:.6g} arcsec"
            )
        else:
            if kappa_i.shape != range_shape or not np.isclose(pk, pix_range, rtol=5e-4):
                raise ValueError(
                    f"Range realization {map_id} uses a different grid: "
                    f"shape={kappa_i.shape}, pix={pk}; expected {range_shape}, {pix_range}"
                )
        mean_kappa += kappa_i
        sum_gamma_obs_for_mean += gamma_i
        if j == 1 or j % 20 == 0 or j == len(pairs):
            print(f"  pass 1: loaded {j:3d}/{len(pairs)}")

    assert mean_kappa is not None and range_header is not None and range_shape is not None and pix_range is not None
    mean_kappa /= float(len(pairs))
    mean_gamma_obs_direct = sum_gamma_obs_for_mean / float(len(pairs))
    mask_range = edge_mask(range_shape, args.edge_fraction)

    # Reconstruct the mean range kappa once.  By linearity, rec_i - rec_mean is
    # the reconstructed field associated with kappa_i - <kappa>.
    rec_mean = reconstruct_from_kappa(
        mean_kappa,
        pix_range,
        args.pad_factor,
        need_potential=args.save_range_psi,
    )

    # Independent best-fit vector-shear anchor resampled to the range WCS.
    gamma1_source_header = gamma1_header if gamma1_header is not None else ref_header
    gamma2_source_header = gamma2_header if gamma2_header is not None else ref_header

    # gamma1/gamma2 are spin-2 components, so we do not silently rotate them
    # between differently oriented image coordinate systems. HFF best-fit/range
    # products are normally aligned; fail explicitly if this is not true.
    ref_orient = wcs_orientation_deg(gamma1_source_header)
    range_orient = wcs_orientation_deg(range_header)
    orient_delta = angle_difference_deg(range_orient, ref_orient)
    print(f"Best-fit/range WCS +x orientation difference: {orient_delta:.6g} deg")
    if abs(orient_delta) > 1e-3:
        raise ValueError(
            "Best-fit and range WCS axes are rotated relative to each other by "
            f"{orient_delta:.6g} deg. This script intentionally does not guess a "
            "spin-2 gamma-component rotation; align the grids or add an explicit "
            "validated shear-basis transformation first."
        )

    gamma1_anchor = resample_wcs(
        gamma1_ref, gamma1_source_header, range_header, range_shape, args.wcs_chunk_rows
    )
    gamma2_anchor = resample_wcs(
        gamma2_ref, gamma2_source_header, range_header, range_shape, args.wcs_chunk_rows
    )

    # Candidate potential anchor uses the already validated best-fit harmonic psi.
    psi_anchor = None
    if args.save_range_psi:
        psi_anchor = resample_wcs(
            rec_ref_h.psi, ref_header, range_header, range_shape, args.wcs_chunk_rows
        )
        if rec_mean.psi is None:
            raise RuntimeError("Internal error: mean range psi missing in save-range-psi mode")
        if np.any(~np.isfinite(psi_anchor)):
            raise ValueError(
                "Best-fit psi does not cover the complete range WCS. Refusing to "
                "write candidate range psi maps with NaN/extrapolated anchor pixels."
            )
        psi_dir = out / "range_psi"
        psi_dir.mkdir(parents=True, exist_ok=True)

    finite_anchor = np.isfinite(gamma1_anchor) & np.isfinite(gamma2_anchor)
    eval_mask = mask_range & finite_anchor
    if np.count_nonzero(eval_mask) < 100:
        raise ValueError("Too few overlapping finite pixels after WCS resampling best-fit gamma anchor")

    # ------------------------------------------------------------------
    # Pass 2: predict every gamma realization from delta-kappa and accumulate
    # sufficient statistics for ensemble-centered comparisons without storing
    # all 100 maps in RAM.
    # ------------------------------------------------------------------
    shape = range_shape
    sum_p = np.zeros(shape, dtype=np.float64)
    sum_o = np.zeros(shape, dtype=np.float64)
    sum_p2 = np.zeros(shape, dtype=np.float64)
    sum_o2 = np.zeros(shape, dtype=np.float64)
    sum_po = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.int32)
    rows: list[dict[str, object]] = []

    for j, (map_id, kp, gp) in enumerate(pairs, start=1):
        kappa_i, kh = read_map(kp)
        gamma_obs, _ = read_map(gp)
        rec_i = reconstruct_from_kappa(
            kappa_i,
            pix_range,
            args.pad_factor,
            need_potential=args.save_range_psi,
        )

        dgamma1 = rec_i.gamma1 - rec_mean.gamma1
        dgamma2 = rec_i.gamma2 - rec_mean.gamma2
        gamma1_pred = gamma1_anchor + dgamma1
        gamma2_pred = gamma2_anchor + dgamma2
        gamma_pred = np.hypot(gamma1_pred, gamma2_pred)

        finite = eval_mask & np.isfinite(gamma_pred) & np.isfinite(gamma_obs)
        if np.count_nonzero(finite) < 100:
            raise ValueError(f"Too few finite comparison pixels in range realization {map_id}")

        # Absolute comparison is secondary: retained to show that the new test is
        # intentionally not trying to optimize away a common absolute mismatch.
        abs_rms = scalar_relative_rms(gamma_pred-gamma_obs, gamma_obs, finite)
        abs_corr = correlation(gamma_pred, gamma_obs, finite)

        p = np.where(finite, gamma_pred, 0.0)
        o = np.where(finite, gamma_obs, 0.0)
        sum_p += p
        sum_o += o
        sum_p2 += p*p
        sum_o2 += o*o
        sum_po += p*o
        count += finite.astype(np.int32)

        rows.append({
            "map_id": map_id,
            "absolute_gamma_rel_rms": abs_rms,
            "absolute_gamma_corr": abs_corr,
            "kappa_closure_rel_rms": scalar_relative_rms(rec_i.kappa-kappa_i, kappa_i, finite),
        })

        if args.save_range_psi:
            if rec_i.psi is None or rec_mean.psi is None or psi_anchor is None:
                raise RuntimeError("Internal error while constructing candidate range psi")
            psi_candidate = psi_anchor + (rec_i.psi - rec_mean.psi)
            out_header = kh.copy()
            out_header["BUNIT"] = "arcsec2"
            out_header.add_history("Candidate diego range psi: validated best-fit anchor + delta-psi from delta-kappa")
            out_header.add_history("Paired scalar gamma used for ensemble-variation validation, not per-map shear fitting")
            out_name = re.sub(r"_kappa\.fits$", "_psi.fits", kp.name)
            fits.PrimaryHDU(psi_candidate.astype(np.float32), header=out_header).writeto(
                psi_dir / out_name, overwrite=True
            )

        if j == 1 or j % 10 == 0 or j == len(pairs):
            print(
                f"  pass 2: {j:3d}/{len(pairs)} map{map_id}: "
                f"absolute |gamma| RMS={abs_rms:.4g}, corr={abs_corr:.4g}"
            )

    count_f = count.astype(np.float64)
    enough = count >= 2
    mean_pred = np.full(shape, np.nan, dtype=np.float64)
    mean_obs = np.full(shape, np.nan, dtype=np.float64)
    mean_pred[count > 0] = sum_p[count > 0] / count_f[count > 0]
    mean_obs[count > 0] = sum_o[count > 0] / count_f[count > 0]

    var_pred = np.full(shape, np.nan, dtype=np.float64)
    var_obs = np.full(shape, np.nan, dtype=np.float64)
    var_pred[enough] = np.maximum(
        sum_p2[enough] / count_f[enough] - mean_pred[enough]**2, 0.0
    )
    var_obs[enough] = np.maximum(
        sum_o2[enough] / count_f[enough] - mean_obs[enough]**2, 0.0
    )
    sigma_pred = np.sqrt(var_pred)
    sigma_obs = np.sqrt(var_obs)

    # Centered ensemble sufficient statistics, summed over realizations at each pixel.
    ss_p = np.zeros(shape, dtype=np.float64)
    ss_o = np.zeros(shape, dtype=np.float64)
    cov_po = np.zeros(shape, dtype=np.float64)
    ss_p[enough] = np.maximum(
        sum_p2[enough] - count_f[enough] * mean_pred[enough]**2, 0.0
    )
    ss_o[enough] = np.maximum(
        sum_o2[enough] - count_f[enough] * mean_obs[enough]**2, 0.0
    )
    cov_po[enough] = (
        sum_po[enough] - count_f[enough] * mean_pred[enough] * mean_obs[enough]
    )
    ss_diff = np.maximum(ss_p + ss_o - 2.0 * cov_po, 0.0)

    valid = eval_mask & enough & np.isfinite(sigma_pred) & np.isfinite(sigma_obs)
    fluct_num = float(np.sqrt(np.sum(ss_diff[valid])))
    fluct_den = float(np.sqrt(np.sum(ss_o[valid])))
    fluct_rel_rms = fluct_num / fluct_den if fluct_den > 0 else np.nan

    cov_total = float(np.sum(cov_po[valid]))
    p_total = float(np.sum(ss_p[valid]))
    o_total = float(np.sum(ss_o[valid]))
    fluct_corr = cov_total / np.sqrt(p_total*o_total) if p_total > 0 and o_total > 0 else np.nan

    # Best linear amplitude mapping of predicted centered fluctuations to observed.
    fluct_slope = cov_total / p_total if p_total > 0 else np.nan

    sigma_rel_rms = scalar_relative_rms(sigma_pred-sigma_obs, sigma_obs, valid)
    sigma_corr = correlation(sigma_pred, sigma_obs, valid)
    sigma_ratio = float(np.sqrt(np.mean(sigma_pred[valid]**2)) / np.sqrt(np.mean(sigma_obs[valid]**2)))

    corr_map = np.full(shape, np.nan, dtype=np.float64)
    denom_map = np.sqrt(ss_p * ss_o)
    good_corr = valid & (denom_map > 0)
    corr_map[good_corr] = cov_po[good_corr] / denom_map[good_corr]
    pixel_corr_median = float(np.nanmedian(corr_map[good_corr])) if np.any(good_corr) else np.nan
    pixel_corr_p16 = float(np.nanpercentile(corr_map[good_corr], 16)) if np.any(good_corr) else np.nan

    # Check the pass-1 direct mean for internal consistency.
    mean_obs_consistency = scalar_relative_rms(mean_obs-mean_gamma_obs_direct, mean_gamma_obs_direct, valid)

    write_csv(out / "range_absolute_diagnostics.csv", rows)

    # Save ensemble diagnostic maps on the native range grid.
    map_header = range_header.copy()
    map_header["BUNIT"] = "dimensionless"
    map_header.add_history("diego ensemble-variation validator output")
    for filename, data, bunit in [
        ("gamma_mean_observed.fits", mean_obs, "dimensionless"),
        ("gamma_mean_predicted.fits", mean_pred, "dimensionless"),
        ("sigma_gamma_observed.fits", sigma_obs, "dimensionless"),
        ("sigma_gamma_predicted.fits", sigma_pred, "dimensionless"),
        ("gamma_fluctuation_correlation.fits", corr_map, "dimensionless"),
    ]:
        h = map_header.copy()
        h["BUNIT"] = bunit
        fits.PrimaryHDU(data.astype(np.float32), header=h).writeto(out / filename, overwrite=True)

    save_ensemble_plot(
        out / "ensemble_variation_validation.png",
        mean_obs,
        mean_pred,
        sigma_obs,
        sigma_pred,
        corr_map,
        valid,
    )
    save_sigma_scatter(out / "sigma_gamma_scatter.png", sigma_obs, sigma_pred, valid, rng)

    # ------------------------------------------------------------------
    # Decision aid: this is intentionally about ensemble variations, not
    # absolute gamma closure.  It still does not approve sigma_dt by itself.
    # ------------------------------------------------------------------
    reasons: list[str] = []
    if not np.isfinite(bestfit_dpsi_rms) or bestfit_dpsi_rms > args.decision_bestfit_dpsi_rms:
        reasons.append(
            f"best-fit harmonic potential-difference RMS {bestfit_dpsi_rms:.4g} > "
            f"{args.decision_bestfit_dpsi_rms:.4g}"
        )
    if not np.isfinite(fluct_rel_rms) or fluct_rel_rms > args.decision_fluct_rms:
        reasons.append(
            f"ensemble fluctuation relative RMS {fluct_rel_rms:.4g} > {args.decision_fluct_rms:.4g}"
        )
    if not np.isfinite(fluct_corr) or fluct_corr < args.decision_fluct_corr:
        reasons.append(
            f"ensemble fluctuation correlation {fluct_corr:.4g} < {args.decision_fluct_corr:.4g}"
        )
    if not np.isfinite(sigma_rel_rms) or sigma_rel_rms > args.decision_sigma_rms:
        reasons.append(
            f"sigma_gamma map relative RMS {sigma_rel_rms:.4g} > {args.decision_sigma_rms:.4g}"
        )
    if not np.isfinite(sigma_corr) or sigma_corr < args.decision_sigma_corr:
        reasons.append(
            f"sigma_gamma map correlation {sigma_corr:.4g} < {args.decision_sigma_corr:.4g}"
        )

    status = "PASS_TO_PSI_ENSEMBLE_STAGE" if not reasons else "DO_NOT_BUILD_SIGMA_DT_YET"

    abs_rms_vals = np.asarray([float(r["absolute_gamma_rel_rms"]) for r in rows])
    abs_corr_vals = np.asarray([float(r["absolute_gamma_corr"]) for r in rows])

    summary = [
        "diego kappa -> psi ensemble-variation validation",
        "=================================================",
        f"diego_dir: {diego_dir}",
        f"range_dir: {range_dir}",
        f"range_realizations: {len(pairs)}",
        f"bestfit_shape: {kappa_ref.shape}",
        f"bestfit_pixscale_arcsec: {pix_ref:.10g}",
        f"range_shape: {range_shape}",
        f"range_pixscale_arcsec: {pix_range:.10g}",
        f"edge_fraction: {args.edge_fraction:g}",
        f"gamma_vector_anchor: {gamma_anchor_source}",
        f"bestfit_range_wcs_orientation_delta_deg: {orient_delta:.10g}",
        f"range_valid_count_min_in_mask: {int(np.min(count[eval_mask]))}",
        f"range_valid_count_max_in_mask: {int(np.max(count[eval_mask]))}",
        "",
        "Best-fit potential anchor validation",
        f"kappa closure relative RMS: {kappa_closure:.10g}",
        f"alpha relative RMS raw: {alpha_raw_rms:.10g}",
        f"alpha relative RMS harmonic: {alpha_h_rms:.10g}",
        f"harmonic bx: {harmonic.bx:.10g} arcsec",
        f"harmonic by: {harmonic.by:.10g} arcsec",
        f"harmonic gamma1: {harmonic.gamma1:.10g}",
        f"harmonic gamma2: {harmonic.gamma2:.10g}",
        f"full-quadratic kappa0 diagnostic: {full_quad.kappa0:.10g}",
    ]
    for name, metrics in line_tests.items():
        summary.append(
            f"potential-difference {name}: rel_rms={metrics['relative_rms']:.10g}, "
            f"median_abs={metrics['median_abs_arcsec2']:.10g} arcsec^2, "
            f"p95_abs={metrics['p95_abs_arcsec2']:.10g} arcsec^2, "
            f"corr={metrics['correlation']:.10g}"
        )

    summary.extend([
        "",
        "Primary ensemble-variation test",
        f"global centered fluctuation relative RMS: {fluct_rel_rms:.10g}",
        f"global centered fluctuation correlation: {fluct_corr:.10g}",
        f"centered fluctuation best-fit amplitude slope (obs vs pred): {fluct_slope:.10g}",
        f"sigma_gamma map relative RMS: {sigma_rel_rms:.10g}",
        f"sigma_gamma map correlation: {sigma_corr:.10g}",
        f"sigma_gamma RMS amplitude ratio pred/obs: {sigma_ratio:.10g}",
        f"per-pixel fluctuation correlation median: {pixel_corr_median:.10g}",
        f"per-pixel fluctuation correlation p16: {pixel_corr_p16:.10g}",
        f"observed-mean accumulator consistency RMS: {mean_obs_consistency:.10g}",
        "",
        "Secondary absolute-|gamma| diagnostic (NOT the decision target)",
        f"absolute gamma relative RMS median: {np.nanmedian(abs_rms_vals):.10g}",
        f"absolute gamma relative RMS p95: {np.nanpercentile(abs_rms_vals,95):.10g}",
        f"absolute gamma correlation median: {np.nanmedian(abs_corr_vals):.10g}",
        "",
        "Decision aid (heuristic; not final sigma_dt validation)",
        f"status: {status}",
        f"best-fit dpsi RMS threshold: {args.decision_bestfit_dpsi_rms:g}",
        f"fluctuation RMS threshold: {args.decision_fluct_rms:g}",
        f"fluctuation correlation threshold: {args.decision_fluct_corr:g}",
        f"sigma_gamma RMS threshold: {args.decision_sigma_rms:g}",
        f"sigma_gamma correlation threshold: {args.decision_sigma_corr:g}",
    ])
    if reasons:
        summary.extend([f"- {r}" for r in reasons])
    else:
        summary.append(
            "- Ensemble gamma-variation checks passed. Proceed to candidate psi-ensemble and "
            "gauge-consistent Fermat-potential/sigma_dt validation; do not yet treat sigma_dt as production-validated."
        )

    summary.extend([
        "",
        "Interpretation",
        "- The primary test compares realization-to-realization deviations, not absolute gamma closure.",
        "- A common finite-field/exterior contribution cancels from ensemble-centered statistics.",
        "- Best-fit gamma1/gamma2 supplies the vector orientation anchor; the range scalar gamma maps are validation targets.",
        "- Candidate psi_i, when requested, equals the validated best-fit harmonic psi anchor plus delta-psi from kappa_i-<kappa>.",
        "- A pass supports proceeding to a separate sigma_dt/Fermat validation; it does not prove that raw pixelwise std(psi) is a valid sigma_dt construction.",
    ])
    (out / "summary.txt").write_text("\n".join(summary) + "\n")

    print("\n=== ENSEMBLE-VARIATION RESULT ===")
    print(f"global centered fluctuation RMS:  {fluct_rel_rms:.6g}")
    print(f"global centered fluctuation corr: {fluct_corr:.6g}")
    print(f"sigma_gamma map RMS:              {sigma_rel_rms:.6g}")
    print(f"sigma_gamma map corr:             {sigma_corr:.6g}")
    print(f"sigma amplitude ratio pred/obs:   {sigma_ratio:.6g}")
    print(f"decision: {status}")
    for r in reasons:
        print(f"  - {r}")

    print(f"\nOutputs written to {out}")
    print("  summary.txt")
    print("  bestfit_validation.png")
    print("  bestfit_psi_harmonic.fits")
    print("  range_absolute_diagnostics.csv")
    print("  ensemble_variation_validation.png")
    print("  sigma_gamma_scatter.png")
    print("  gamma_mean_observed.fits / gamma_mean_predicted.fits")
    print("  sigma_gamma_observed.fits / sigma_gamma_predicted.fits")
    print("  gamma_fluctuation_correlation.fits")
    if args.save_range_psi:
        print("  range_psi/*.fits")


if __name__ == "__main__":
    main()