#!/usr/bin/env python3
"""
Simplified conditional estimate of R_nondet for the six Hubble
Frontier Fields (HFF) cluster sightlines.

Estimator
---------
                  Omega_HFF ∫ dz dlnM (dn/dlnM) (dVc/dz/dOmega)
                              sigma_SL L_miss (1 - C_cat)
R_nondet  =  ----------------------------------------------------------------
                         Σ_i sigma_SL,i L_i

This is conditional on the lens lying in one of the six HFF cluster
sightlines. It is not an all-sky missing-cluster probability.

The code uses:
* Astropy Planck18 for cosmological distances, volume, densities,
  physical constants, and units.
* Colossus for the Tinker et al. (2008) M_200m halo mass function.
* An SIS approximation for strong-lensing cross-sections.

Install
-------
pip install astropy colossus scipy numpy matplotlib
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.constants import G, c
from astropy.cosmology import Planck18
from colossus.cosmology import cosmology as colossus_cosmology
from colossus.lss import mass_function
from scipy.integrate import cumulative_trapezoid, trapezoid


# =============================================================================
# Configuration
# =============================================================================

COSMO = Planck18
SIGMA8 = 0.8102
NS = 0.9665

SOURCE_REDSHIFT = 3.0

# Missing-halo integration range, interpreted as physical M_200m masses.
BASELINE_MISSING_MASS_MIN_MSUN = 1.0e15
MASS_GRID_MIN_MSUN = 2.0e14
MASS_GRID_MAX_MSUN = 5.0e15

N_MASS = 420
N_REDSHIFT = 320

# Six nominal ACS/WFC cluster-core footprints
ACS_WFC_SIDE = 202.0 * u.arcsec

# Conservative upper estimate for the missing branch:
# C_cat = 0 counts every additional massive halo as uncatalogued.
MISSING_CATALOG_COMPLETENESS = 0.0

# Population-only baseline: the actual GW data do not yet discriminate
# between known and missing cluster lenses.
KNOWN_GW_LIKELIHOODS = np.ones(6)

# Plot output directory.
PLOT_OUTPUT_DIR = Path("r_nondet_plots")
PLOT_DPI = 220


# Approximate placeholders for the denominator.
# These are treated as M_200m only for this toy calculation.
# Replace them with consistently converted masses or, preferably,
# cross-sections measured directly from the public HFF lens maps.
HFF_NAMES = np.array(
    [
        "Abell 2744",
        "MACS J0416.1-2403",
        "MACS J0717.5+3745",
        "MACS J1149.5+2223",
        "Abell S1063",
        "Abell 370",
    ]
)

HFF_REDSHIFTS = np.array(
    [0.308, 0.396, 0.545, 0.543, 0.348, 0.375],
    dtype=float,
)

HFF_M200M_MSUN = np.array(
    [1.8e15, 1.2e15, 2.5e15, 2.5e15, 1.4e15, 1.0e15],
    dtype=float,
)


# Make the Colossus HMF use the same background cosmology as Astropy.
# Astropy does not contain sigma_8 or n_s, so they are supplied explicitly.
COLOSSUS_COSMO = colossus_cosmology.fromAstropy(
    COSMO,
    sigma8=SIGMA8,
    ns=NS,
    cosmo_name="hff_planck18",
    print_warnings=False,
)


# =============================================================================
# Selection and GW-likelihood placeholders
# =============================================================================

def catalog_completeness(
    mass_msun: np.ndarray,
    redshift: float,
) -> np.ndarray:
    """
    C_cat(M,z) = P(the additional halo is represented in the EM catalog).

    The baseline C_cat=0 is deliberately conservative: it maximizes
    Z_missing. Replace this with a calibrated mass-redshift completeness
    function when available.
    """
    del redshift

    return np.full_like(
        mass_msun,
        MISSING_CATALOG_COMPLETENESS,
        dtype=float,
    )


def missing_gw_likelihood(
    mass_msun: np.ndarray,
    redshift: float,
) -> np.ndarray:
    """
    L_miss(M,z) = P(d_GW | M,z, missing cluster, H_GW^L).

    The population-only baseline sets this to one. Replace it with an
    event-specific likelihood based on time delays, magnification ratios,
    image multiplicity, luminosity distance, or the full GW data.
    """
    del redshift
    return np.ones_like(mass_msun, dtype=float)


# =============================================================================
# SIS lensing cross-section
# =============================================================================

def sis_einstein_radius(
    mass_m200m_msun: np.ndarray,
    lens_redshift: np.ndarray | float,
    source_redshift: float,
) -> np.ndarray:
    """
    Approximate an M_200m halo as a singular isothermal sphere.

    Returns
    -------
    theta_e : ndarray
        Einstein radius in radians.
    """
    mass = np.asarray(mass_m200m_msun, dtype=float) * u.Msun
    z_l = np.asarray(lens_redshift, dtype=float)

    # Physical mean matter density at z_l.
    rho_m_z = (
        COSMO.Om(z_l) * COSMO.critical_density(z_l)
    ).to(u.Msun / u.Mpc**3)

    # M_200m = (4π/3) 200 rho_m(z) r_200m^3.
    r_200m = (
        3.0 * mass
        / (4.0 * np.pi * 200.0 * rho_m_z)
    ) ** (1.0 / 3.0)

    # Match the SIS enclosed mass M(<r)=2 sigma_v^2 r/G at r_200m.
    sigma_v = np.sqrt(
        (G * mass / (2.0 * r_200m)).to(u.km**2 / u.s**2)
    )

    d_s = COSMO.angular_diameter_distance(source_redshift)

    d_ls = COSMO.angular_diameter_distance_z1z2(
        z_l,
        source_redshift,
    )

    theta_e = (
        4.0
        * np.pi
        * (sigma_v / c) ** 2
        * d_ls
        / d_s
    ).decompose().value

    return theta_e


def sis_cross_section_sr(
    mass_m200m_msun: np.ndarray,
    lens_redshift: np.ndarray | float,
    source_redshift: float,
) -> np.ndarray:
    """
    Angular source-plane multiple-imaging cross-section:
        sigma_SL = π theta_E^2.
    """
    theta_e = sis_einstein_radius(
        mass_m200m_msun,
        lens_redshift,
        source_redshift,
    )
    return np.pi * theta_e**2


# =============================================================================
# HMF and evidence-like weights
# =============================================================================

def halo_mass_function_dndlnm(
    mass_m200m_msun: np.ndarray,
    redshift: float,
) -> np.ndarray:
    """
    Tinker08 dn/dlnM in physical comoving Mpc^-3.
    Colossus input masses are numerical values in Msun/h and its dndlnM
    output has units (Mpc/h)^-3.
    """
    h = COSMO.h

    # Physical Msun -> numerical Msun/h used by Colossus.
    mass_msun_over_h = mass_m200m_msun * h

    dndlnm_mpc_over_h = mass_function.massFunction(
        mass_msun_over_h,
        redshift,
        q_in="M",
        q_out="dndlnM",
        mdef="200m",
        model="tinker08",
    )

    # (Mpc/h)^-3 = h^3 Mpc^-3.
    return dndlnm_mpc_over_h * h**3


def hff_solid_angle_sr() -> float:
    side_radians = ACS_WFC_SIDE.to_value(u.rad)
    return 6 * side_radians**2


def known_hff_weight() -> tuple[float, np.ndarray]:
    """
    Z_known = Σ_i sigma_SL,i L_i.
    """
    cross_sections = sis_cross_section_sr(
        HFF_M200M_MSUN,
        HFF_REDSHIFTS,
        SOURCE_REDSHIFT,
    )

    z_known = np.sum(
        cross_sections * KNOWN_GW_LIKELIHOODS
    )

    return float(z_known), cross_sections


def build_missing_integrand() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Construct

      I(M,z) = (dn/dlnM)
               (dVc/dz/dOmega)
               sigma_SL
               L_miss
               (1-C_cat).

    Returns
    -------
    mass_msun, redshift, integrand
    """
    mass_msun = np.geomspace(
        MASS_GRID_MIN_MSUN,
        MASS_GRID_MAX_MSUN,
        N_MASS,
    )

    redshift = np.linspace(
        0.01,
        SOURCE_REDSHIFT - 0.01,
        N_REDSHIFT,
    )

    integrand = np.empty(
        (redshift.size, mass_msun.size),
        dtype=float,
    )

    for iz, z_l in enumerate(redshift):
        dndlnm = halo_mass_function_dndlnm(
            mass_msun,
            z_l,
        )

        dvc_dz_domega = (
            COSMO.differential_comoving_volume(z_l)
            .to_value(u.Mpc**3 / u.sr)
        )

        sigma_sl = sis_cross_section_sr(
            mass_msun,
            z_l,
            SOURCE_REDSHIFT,
        )

        likelihood = missing_gw_likelihood(
            mass_msun,
            z_l,
        )

        completeness = catalog_completeness(
            mass_msun,
            z_l,
        )

        integrand[iz] = (
            dndlnm
            * dvc_dz_domega
            * sigma_sl
            * likelihood
            * (1.0 - completeness)
        )

    return mass_msun, redshift, integrand


def missing_weight_above_mass(
    mass_msun: np.ndarray,
    redshift: np.ndarray,
    integrand: np.ndarray,
    minimum_mass_msun: float,
) -> float:
    """
    Z_missing = Omega_HFF ∫ dz ∫_{M_min} dlnM I(M,z).
    """
    selected = mass_msun >= minimum_mass_msun

    if np.count_nonzero(selected) < 2:
        raise ValueError(
            "The selected mass interval contains fewer than two grid points."
        )

    integral_over_ln_mass = trapezoid(
        integrand[:, selected],
        x=np.log(mass_msun[selected]),
        axis=1,
    )

    integral_over_redshift = trapezoid(
        integral_over_ln_mass,
        x=redshift,
    )

    return float(
        hff_solid_angle_sr()
        * integral_over_redshift
    )


# =============================================================================
# Plotting
# =============================================================================

def cumulative_r_nondet_curve(
    mass_msun: np.ndarray,
    redshift: np.ndarray,
    integrand: np.ndarray,
    z_known: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute R_nondet(>M_min) for every mass-grid point without rebuilding
    the HMF integrand.

    For each redshift, integrate the missing-lens contribution from the
    current mass grid point to the maximum mass. Then integrate over z and
    divide by Z_known.
    """
    ln_mass = np.log(mass_msun)

    # Reverse the mass axis so cumulative_trapezoid integrates from high
    # mass downward. The minus sign corrects for the decreasing x order.
    cumulative_from_high_mass_reversed = -cumulative_trapezoid(
        integrand[:, ::-1],
        x=ln_mass[::-1],
        axis=1,
        initial=0.0,
    )

    cumulative_from_each_mass = (
        cumulative_from_high_mass_reversed[:, ::-1]
    )

    z_missing_above_mass = (
        hff_solid_angle_sr()
        * trapezoid(
            cumulative_from_each_mass,
            x=redshift,
            axis=0,
        )
    )

    return mass_msun, z_missing_above_mass / z_known


def plot_missing_integrand_map(
    mass_msun: np.ndarray,
    redshift: np.ndarray,
    integrand: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot the differential missing-lens weight

      d^2 Z_missing / (dz dlnM) = Omega_HFF * I(M,z).

    The six known HFF clusters and the baseline missing-mass threshold
    are overlaid for context.
    """
    differential_weight_sr = (
        hff_solid_angle_sr() * integrand
    )

    positive = differential_weight_sr[
        differential_weight_sr > 0.0
    ]
    if positive.size == 0:
        raise RuntimeError(
            "The missing-lens integrand contains no positive values."
        )

    log_weight = np.full_like(
        differential_weight_sr,
        np.nan,
        dtype=float,
    )
    log_weight[differential_weight_sr > 0.0] = np.log10(
        differential_weight_sr[differential_weight_sr > 0.0]
    )

    figure, axis = plt.subplots(figsize=(7.2, 5.2))

    mesh = axis.pcolormesh(
        redshift,
        np.log10(mass_msun),
        log_weight.T,
        shading="auto",
    )

    colorbar = figure.colorbar(mesh, ax=axis)
    colorbar.set_label(
        r"$\log_{10}\!\left[d^2 Z_{\rm miss}/"
        r"(dz_l\,d\ln M)\,/\,{\rm sr}\right]$"
    )

    axis.scatter(
        HFF_REDSHIFTS,
        np.log10(HFF_M200M_MSUN),
        marker="o",
        label="Known HFF clusters",
    )

    for name, z_l, mass in zip(
        HFF_NAMES,
        HFF_REDSHIFTS,
        HFF_M200M_MSUN,
        strict=True,
    ):
        axis.annotate(
            name,
            (z_l, np.log10(mass)),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    axis.axhline(
        np.log10(BASELINE_MISSING_MASS_MIN_MSUN),
        linestyle="--",
        label=(
            r"Baseline $M_{\rm min}=10^{15}\,M_\odot$"
        ),
    )

    axis.set_xlabel(r"$z_l$")
    axis.set_ylabel(r"$\log_{10}(M_{200{\rm m}}/M_\odot)$")
    axis.set_title("Weight of missing clusters")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(figure)


def plot_r_nondet_mass_cut(
    mass_cuts_msun: np.ndarray,
    r_nondet_values: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot the cumulative ratio R_nondet(>M_min).
    """
    valid = (
        np.isfinite(r_nondet_values)
        & (r_nondet_values > 0.0)
    )

    figure, axis = plt.subplots(figsize=(7.2, 5.0))

    axis.loglog(
        mass_cuts_msun[valid],
        r_nondet_values[valid],
        linewidth=2.0,
        label=r"$R_{\rm nondet}(>M_{\rm min})$",
    )

    axis.axhline(
        1.0e-4,
        linestyle=":",
        label=r"$R_{\rm nondet}=10^{-4}$",
    )
    axis.axvline(
        BASELINE_MISSING_MASS_MIN_MSUN,
        linestyle="--",
        label=r"$M_{\rm min}=10^{15}\,M_\odot$",
    )

    baseline_index = int(
        np.argmin(
            np.abs(
                np.log(mass_cuts_msun)
                - np.log(BASELINE_MISSING_MASS_MIN_MSUN)
            )
        )
    )

    axis.scatter(
        [mass_cuts_msun[baseline_index]],
        [r_nondet_values[baseline_index]],
        marker="o",
        zorder=3,
    )
    axis.annotate(
        (
            rf"$R={r_nondet_values[baseline_index]:.1e}$"
        ),
        (
            mass_cuts_msun[baseline_index],
            r_nondet_values[baseline_index],
        ),
        xytext=(8, 8),
        textcoords="offset points",
    )

    axis.set_xlabel(
        r"Minimum mass of an missed lens "
        r"$M_{\rm min}\,[M_\odot]$"
    )
    axis.set_ylabel(r"$R_{\rm nondet}(>M_{\rm min})$")
    axis.set_title(
        "Sensitivity to the minimum missing-cluster mass"
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(figure)


def save_weekly_update_plots(
    mass_msun: np.ndarray,
    redshift: np.ndarray,
    integrand: np.ndarray,
    z_known: float,
) -> tuple[Path, Path]:
    """
    Save the two weekly-update figures from the same arrays used in the
    R_nondet calculation.
    """
    PLOT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    integrand_path = (
        PLOT_OUTPUT_DIR
        / "missing_lens_integrand_mass_redshift.png"
    )
    mass_cut_path = (
        PLOT_OUTPUT_DIR
        / "r_nondet_vs_minimum_mass.png"
    )

    plot_missing_integrand_map(
        mass_msun,
        redshift,
        integrand,
        integrand_path,
    )

    mass_cuts, r_values = cumulative_r_nondet_curve(
        mass_msun,
        redshift,
        integrand,
        z_known,
    )

    plot_r_nondet_mass_cut(
        mass_cuts,
        r_values,
        mass_cut_path,
    )

    return integrand_path, mass_cut_path


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    z_known, known_cross_sections = known_hff_weight()

    mass_msun, redshift, missing_integrand = (
        build_missing_integrand()
    )

    z_missing = missing_weight_above_mass(
        mass_msun,
        redshift,
        missing_integrand,
        BASELINE_MISSING_MASS_MIN_MSUN,
    )

    r_nondet = z_missing / z_known

    theta_e_arcsec = (
        sis_einstein_radius(
            HFF_M200M_MSUN,
            HFF_REDSHIFTS,
            SOURCE_REDSHIFT,
        )
        * u.rad
    ).to_value(u.arcsec)

    print("Conditional HFF R_nondet estimate")
    print("---------------------------------")
    print(f"Source redshift:              {SOURCE_REDSHIFT:.3f}")
    print(f"Six-field solid angle:        {hff_solid_angle_sr():.6e} sr")
    print(
        "Baseline missing mass cut:    "
        f"{BASELINE_MISSING_MASS_MIN_MSUN:.3e} Msun"
    )
    print(
        "Missing completeness C_cat:   "
        f"{MISSING_CATALOG_COMPLETENESS:.3f}"
    )
    print()

    print("Approximate known HFF SIS lenses")
    print("--------------------------------")
    for name, z_l, mass, theta_e, cross_section in zip(
        HFF_NAMES,
        HFF_REDSHIFTS,
        HFF_M200M_MSUN,
        theta_e_arcsec,
        known_cross_sections,
        strict=True,
    ):
        print(
            f"{name:22s} "
            f"z={z_l:.3f} "
            f"M200m={mass:.2e} Msun "
            f"theta_E={theta_e:5.2f} arcsec "
            f"sigma_SL={cross_section:.3e} sr"
        )

    print()
    print("Baseline result")
    print("---------------")
    print(f"Z_known:                     {z_known:.6e} sr")
    print(f"Z_missing:                   {z_missing:.6e} sr")
    print(f"R_nondet:                    {r_nondet:.6e}")
    print(f"log10(R_nondet):             {np.log10(r_nondet):.3f}")

    print()
    print("Mass-cut sensitivity")
    print("--------------------")
    for mass_cut in (
        2.0e14,
        5.0e14,
        8.0e14,
        1.0e15,
        1.5e15,
        2.0e15,
    ):
        trial_z_missing = missing_weight_above_mass(
            mass_msun,
            redshift,
            missing_integrand,
            mass_cut,
        )
        print(
            f"M_min={mass_cut:.2e} Msun  "
            f"R_nondet={trial_z_missing / z_known:.3e}"
        )

    integrand_plot, mass_cut_plot = save_weekly_update_plots(
        mass_msun,
        redshift,
        missing_integrand,
        z_known,
    )

    print()
    print("Saved plots")
    print("-----------")
    print(integrand_plot)
    print(mass_cut_plot)


if __name__ == "__main__":
    main()