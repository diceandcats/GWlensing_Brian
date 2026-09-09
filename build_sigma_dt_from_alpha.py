#!/usr/bin/env python3
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import map_coordinates


# ------------------------------------------------------------------
# Hard-coded production configuration
# ------------------------------------------------------------------

BASE = Path("/home/dices/Research/GWlensing_Brian/GCdata")
MODEL = "diego"
VERSION = "v4"
EDGE = 0.15
SIGMA_SCALE = 1.0
WCS_CHUNK = 128

CLUSTERS = {
    "abell370": BASE / "Abell 370" / "diego",
    "abell2744": BASE / "Abell 2744" / "diego",
    "abells1063": BASE / "Abell S1063" / "diego",
    "macs0416": BASE / "MACS J0416.1-2403" / "diego",
    "macs0717": BASE / "MACS J0717.5+3745" / "diego",
    "macs1149": BASE / "MACS J1149.5+2223" / "diego",
}


def read(path):
    with fits.open(path, memmap=True) as h:
        return np.asarray(h[0].data, dtype=np.float64), h[0].header.copy()


def pixscale(header):
    s = np.asarray(proj_plane_pixel_scales(WCS(header).celestial)) * 3600.0
    if len(s) != 2 or not np.all(np.isfinite(s)) or not np.isclose(s[0], s[1], rtol=5e-3):
        raise ValueError(f"Invalid/non-square pixel scale: {s}")
    return float(np.mean(s))


def mask(shape):
    ny, nx = shape
    ey, ex = int(EDGE * ny), int(EDGE * nx)
    m = np.zeros(shape, bool)
    m[ey:ny-ey, ex:nx-ex] = True
    return m


def cumtrapz(a, d, axis):
    out = np.zeros_like(a)
    if axis == 1:
        out[:, 1:] = np.cumsum(0.5 * (a[:, :-1] + a[:, 1:]) * d, axis=1)
    else:
        out[1:, :] = np.cumsum(0.5 * (a[:-1, :] + a[1:, :]) * d, axis=0)
    return out


def psi_from_alpha(ax, ay, d, m):
    """Symmetric finite-domain integration of alpha = grad(psi)."""
    ny, nx = ax.shape
    ey, ex = int(EDGE * ny), int(EDGE * nx)

    # x-first: psi = integral(alpha_x dx) + C(y)
    px = cumtrapz(ax, d, 1)
    cy_prime = np.median(
        ay[:, ex:nx-ex] - np.gradient(px, d, axis=0, edge_order=2)[:, ex:nx-ex],
        axis=1,
    )
    cy = np.zeros(ny)
    cy[1:] = np.cumsum(0.5 * (cy_prime[:-1] + cy_prime[1:]) * d)
    px += cy[:, None]
    px -= np.median(px[m])

    # y-first: psi = integral(alpha_y dy) + D(x)
    py = cumtrapz(ay, d, 0)
    dx_prime = np.median(
        ax[ey:ny-ey, :] - np.gradient(py, d, axis=1, edge_order=2)[ey:ny-ey, :],
        axis=0,
    )
    dx = np.zeros(nx)
    dx[1:] = np.cumsum(0.5 * (dx_prime[:-1] + dx_prime[1:]) * d)
    py += dx[None, :]
    py -= np.median(py[m])

    psi = 0.5 * (px + py)
    psi -= np.median(psi[m])

    gy, gx = np.gradient(psi, d, d, edge_order=2)
    grad_rms = np.sqrt(
        np.mean((gx[m] - ax[m])**2 + (gy[m] - ay[m])**2)
        / np.mean(ax[m]**2 + ay[m]**2)
    )

    dax_dy, dax_dx = np.gradient(ax, d, d, edge_order=2)
    day_dy, day_dx = np.gradient(ay, d, d, edge_order=2)
    curl = day_dx - dax_dy
    deriv = dax_dx**2 + dax_dy**2 + day_dx**2 + day_dy**2
    curl_rel = np.sqrt(np.mean(curl[m]**2) / np.mean(deriv[m]))

    return psi, float(grad_rms), float(curl_rel)


def orientation(header):
    a = np.asarray(WCS(header).celestial.pixel_scale_matrix)
    return np.degrees(np.arctan2(a[1, 0], a[0, 0]))


def resample(data, src_header, dst_header, dst_shape, mode="constant"):
    sw, dw = WCS(src_header).celestial, WCS(dst_header).celestial
    ny, nx = dst_shape
    out = np.empty(dst_shape)
    xp = np.arange(nx, dtype=float)
    cval = np.nan if mode == "constant" else 0.0

    for y0 in range(0, ny, WCS_CHUNK):
        y1 = min(ny, y0 + WCS_CHUNK)
        yy, xx = np.meshgrid(np.arange(y0, y1, dtype=float), xp, indexing="ij")
        lon, lat = dw.pixel_to_world_values(xx, yy)
        sx, sy = sw.world_to_pixel_values(lon, lat)
        out[y0:y1] = map_coordinates(
            data, [sy.ravel(), sx.ravel()],
            order=1, mode=mode, cval=cval, prefilter=False
        ).reshape(y1-y0, nx)
    return out


def range_pairs(range_dir):
    pairs = []
    for xp in sorted(range_dir.rglob("*x-arcsec-deflect*.fits")):
        yp = xp.with_name(xp.name.replace("x-arcsec-deflect", "y-arcsec-deflect"))
        if yp.is_file():
            pairs.append((xp, yp))
    if len(pairs) < 2:
        raise FileNotFoundError(
            f"Need realization-level x/y deflection maps under {range_dir}; found {len(pairs)} pairs"
        )
    return pairs


def run(cluster, directory):
    stem = f"hlsp_frontier_model_{cluster}_{MODEL}_{VERSION}"
    best_x = directory / f"{stem}_x-arcsec-deflect.fits"
    best_y = directory / f"{stem}_y-arcsec-deflect.fits"
    out = directory / f"{stem}_sigma_dt.fits"
    pairs = range_pairs(directory / "range")

    ax0, h0 = read(best_x)
    ay0, hy0 = read(best_y)
    if ax0.shape != ay0.shape:
        raise ValueError(f"{cluster}: best-fit x/y shapes differ")
    d0 = pixscale(h0)
    m0 = mask(ax0.shape)

    # Best-fit psi is only an internal anchor; it is not written.
    psi0, grad0, curl0 = psi_from_alpha(ax0, ay0, d0, m0)
    if grad0 > 0.05 or curl0 > 0.10:
        raise RuntimeError(
            f"{cluster}: best-fit alpha->psi failed: gradRMS={grad0:.4g}, curlRel={curl0:.4g}"
        )

    # Pass 1: ensemble mean alpha on the native realization grid.
    mean_ax = mean_ay = hr = None
    dr = None
    shape = None

    for i, (xp, yp) in enumerate(pairs):
        ax, hx = read(xp)
        ay, hy = read(yp)
        if ax.shape != ay.shape:
            raise ValueError(f"{cluster}: range x/y shapes differ in {xp.name}")
        d = pixscale(hx)

        if i == 0:
            shape, hr, dr = ax.shape, hx, d
            mean_ax = np.zeros(shape)
            mean_ay = np.zeros(shape)
        elif ax.shape != shape or not np.isclose(d, dr, rtol=5e-4):
            raise ValueError(f"{cluster}: inconsistent realization grids")

        mean_ax += ax
        mean_ay += ay

    mean_ax /= len(pairs)
    mean_ay /= len(pairs)

    # x/y components can be directly resampled only if the map axes are aligned.
    delta_angle = (orientation(hr) - orientation(h0) + 180.0) % 360.0 - 180.0
    if abs(delta_angle) > 1e-3:
        raise ValueError(f"{cluster}: best/range WCS rotated by {delta_angle:.6g} deg")

    mr = mask(shape)
    psi_anchor = resample(psi0, h0, hr, shape)
    ax_anchor = resample(ax0, h0, hr, shape)
    ay_anchor = resample(ay0, hy0, hr, shape)
    finite = np.isfinite(psi_anchor) & np.isfinite(ax_anchor) & np.isfinite(ay_anchor)
    if np.any(~finite[mr]):
        raise ValueError(f"{cluster}: best-fit maps do not cover central range grid")

    q_anchor = 0.5 * (ax_anchor**2 + ay_anchor**2) - psi_anchor

    # Pass 2: gauge-align each realization deviation, reconstruct delta-psi,
    # build Q_i, and accumulate the sample variance online.
    q_mean = np.zeros(shape)
    q_m2 = np.zeros(shape)
    grad_errors = []
    curl_errors = []

    for n, (xp, yp) in enumerate(pairs, 1):
        ax, _ = read(xp)
        ay, _ = read(yp)

        dax = ax - mean_ax
        day = ay - mean_ay

        # Remove constant-deflection / linear-potential source-plane gauge.
        dax -= np.median(dax[mr])
        day -= np.median(day[mr])

        dpsi, grad, curl = psi_from_alpha(dax, day, dr, mr)
        grad_errors.append(grad)
        curl_errors.append(curl)

        qi = (
            0.5 * ((ax_anchor + dax)**2 + (ay_anchor + day)**2)
            - (psi_anchor + dpsi)
        )

        delta = qi - q_mean
        q_mean += delta / n
        q_m2 += delta * (qi - q_mean)

    sigma_q = np.sqrt(np.maximum(q_m2 / (len(pairs) - 1), 0.0))

    valid = mr & finite & np.isfinite(q_anchor)
    scale = np.median(np.abs(q_anchor[valid]))
    denom = np.maximum(np.abs(q_anchor), 1e-6 * scale)
    sigma_dt = SIGMA_SCALE * sigma_q / denom

    # Only output produced by this script.
    sigma_best = resample(sigma_dt, hr, h0, ax0.shape, mode="nearest")
    hout = h0.copy()
    hout["BUNIT"] = "dimensionless"
    hout.add_history("sigma_dt from realization-level x/y deflection Fermat scatter")
    hout.add_history("psi reconstructed internally from alpha=grad(psi); psi files not written")
    hout.add_history("constant deflection gauge removed from each realization deviation")
    fits.PrimaryHDU(
        np.asarray(sigma_best, dtype=np.float32), header=hout
    ).writeto(out, overwrite=True)

    vals = sigma_dt[valid]
    print(
        f"{cluster}: N={len(pairs)}, "
        f"best gradRMS={grad0:.4g}, best curlRel={curl0:.4g}, "
        f"range gradRMS median={np.median(grad_errors):.4g}, "
        f"range curlRel median={np.median(curl_errors):.4g}, "
        f"sigma_dt median={np.median(vals):.4g}, p95={np.percentile(vals,95):.4g}"
    )
    print(f"  wrote {out}")


for cluster, directory in CLUSTERS.items():
    run(cluster, directory)