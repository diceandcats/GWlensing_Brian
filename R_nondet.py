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
sightlines.

The code uses:
- Astropy Planck18 cosmology for distances and volumes.
- Colossus for the Tinker et al. (2008) M_200m halo mass function.
- An SIS approximation for strong-lensing cross-sections.
"""

from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.constants import G, c
from astropy.cosmology import Planck18
from colossus.cosmology import cosmology as colossus_cosmology
from colossus.lss import mass_function
from scipy.integrate import trapezoid


# Configuration

COSMO = Planck18
SIGMA8 = 0.8102
NS = 0.9665

SOURCE_REDSHIFT = 2.0

# Missing-halo integration range, interpreted as physical M_200m masses.
BASELINE_MISSING_MASS_MIN_MSUN = 1.0e15
MASS_GRID_MIN_MSUN = 2.0e14
MASS_GRID_MAX_MSUN = 5.0e15

N_MASS = 420
N_REDSHIFT = 320

# Six nominal ACS/WFC cluster-core footprints.
N_HFF_FIELDS = 6
ACS_WFC_SIDE = 202.0 * u.arcsec

# Conservative upper estimate for the missing branch:
# C_cat = 0 counts every additional massive halo as uncatalogued.
MISSING_CATALOG_COMPLETENESS = 0.0

# Population-only baseline: the actual GW data do not yet discriminate
# between known and missing cluster lenses.
KNOWN_GW_LIKELIHOODS = np.ones(6)


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

    # Astropy >= 8.0: two arguments return D_A(z_l, z_s).
    d_ls = COSMO.angular_diameter_distance(
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
    return N_HFF_FIELDS * side_radians**2


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


if __name__ == "__main__":
    main()