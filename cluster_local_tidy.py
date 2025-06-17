from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Sequence
from functools import lru_cache

import numpy as np
from astropy.cosmology import FlatLambdaCDM  # heavy ‑> cache objects
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from scipy.optimize import differential_evolution
import emcee

__all__ = [
    "ClusterData",
    "ClusterLensUtils",
    "ClusterLensingCore",
]

# -----------------------------------------------------------------------------
# 1. Pure‑data container (fast __init__, hashable when needed)
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class ClusterData:
    """Light-weight immutable container for lensing maps & meta-data."""

    alpha_x: Sequence[np.ndarray]
    alpha_y: Sequence[np.ndarray]
    potential: Sequence[np.ndarray]
    pixscale: Sequence[float]

    z_l: Sequence[float]
    x_center: Sequence[float]
    y_center: Sequence[float]
    search_window: Sequence[float]

    # optional: original source z for pre‑built LensModel objects
    z_source_default: float = 2.0
    H0_default: float = 70.0

    def __post_init__(self):
        n = len(self.alpha_x)
        assert n == len(self.alpha_y) == len(self.potential) == len(self.pixscale), "Inconsistent cluster list lengths"
        self.size = tuple(len(a) for a in self.alpha_x)


# -----------------------------------------------------------------------------
# 2. Utility mix‑in with *no* cluster‑state; safe to call from multiprocessing.
# -----------------------------------------------------------------------------
class ClusterLensUtils:
    """Stateless helpers - kept outside the hot loop whenever possible."""

    @staticmethod
    @lru_cache(maxsize=None)
    def cosmology(H0: float) -> FlatLambdaCDM:  # heavy construction cached
        """Return a cached cosmology object for the given H0."""
        return FlatLambdaCDM(H0=H0, Om0=0.3)

    # small helper: D_LS / D_S scale factor
    @staticmethod
    def scale_factor(cosmo: FlatLambdaCDM, z_l: float, z_s: float) -> float:
        """Compute the scale factor D_LS / D_S for lensing equations."""
        D_S = cosmo.angular_diameter_distance(z_s)
        D_LS = cosmo.angular_diameter_distance_z1z2(z_l, z_s)
        return (D_LS / D_S) if (D_S != 0 and D_LS != 0) else 1.0

    # build LensModel+Solver only once per unique (z_s, H0)
    @staticmethod
    @lru_cache(maxsize=None)
    def lens_solver(z_s: float, z_l: float, H0: float):
        """Return a cached LensEquationSolver for the given z_s, z_l, H0."""
        cosmo = ClusterLensUtils.cosmology(H0)
        lm = LensModel(lens_model_list=["INTERPOL"], z_source=z_s, z_lens=z_l, cosmo=cosmo, cosmology_model=None)
        return LensEquationSolver(lm)


# -----------------------------------------------------------------------------
# 3. Main algorithmic class
# -----------------------------------------------------------------------------
class ClusterLensingCore(ClusterLensUtils):
    """All heavy computations; accepts a :class:`ClusterData` bundle at init."""

    def __init__(self, data: ClusterData):
        self.data = data
        # Pre‑bake per‑cluster static grids & keyword dicts
        self._precompute_maps()

    # ------------------------------------------------------------------
    # Pre‑computation (executed once) – nothing here is touched per χ².
    # ------------------------------------------------------------------
    def _precompute_maps(self):
        self._grid_cache: List[Dict[str, np.ndarray]] = []
        for size, pix, pot, ax, ay in zip(
            self.data.size,
            self.data.pixscale,
            self.data.potential,
            self.data.alpha_x,
            self.data.alpha_y,
        ):
            x_grid = np.linspace(0, size - 1, size) * pix
            self._grid_cache.append(
                {
                    "x_grid": x_grid,
                    "pix2": pix ** 2,
                    "alpha_x0": ax,
                    "alpha_y0": ay,
                    "potential0": pot,
                }
            )

    # ------------------------------------------------------------------
    # Core Low‑level ops (used by higher‑level algorithms) ---------------
    # ------------------------------------------------------------------
    def _candidate_kwargs(self, index: int, z_s: float, H0: float) -> Dict[str, np.ndarray]:
        cache = self._grid_cache[index]
        cosmo = self.cosmology(H0)
        scale = self.scale_factor(cosmo, self.data.z_l[index], z_s)
        return {
            "grid_interp_x": cache["x_grid"],
            "grid_interp_y": cache["x_grid"],
            "f_": cache["potential0"] * scale * cache["pix2"],
            "f_x": cache["alpha_x0"] * scale,
            "f_y": cache["alpha_y0"] * scale,
        }

    def image_positions(
        self,
        x_src: float,
        y_src: float,
        z_s: float,
        *,
        index: int = 0,
        H0: float | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vector-compatible wrapper; *no branches in hot loop*."""
        if H0 is None:
            H0 = self.data.H0_default
        solver = self.lens_solver(z_s, self.data.z_l[index], H0)
        kwargs_lens = [self._candidate_kwargs(index, z_s, H0)]
        return solver.image_position_from_source(
            x_src,
            y_src,
            kwargs_lens,
            min_distance=self.data.pixscale[index],
            search_window=self.data.search_window[index],
            verbose=False,
            x_center=self.data.x_center[index],
            y_center=self.data.y_center[index],
        )

    def time_delays(    
        self,
        x_img: np.ndarray,
        y_img: np.ndarray,
        x_src: float,
        y_src: float,
        z_s: float,
        *,
        index: int = 0,
        H0: float | None = None,
    ) -> np.ndarray:
        """Compute time delays for given image positions and source."""
        if H0 is None:
            H0 = self.data.H0_default
        solver = self.lens_solver(z_s, self.data.z_l[index], H0)
        lm = solver.lensModel  # camel-case attribute in lenstronomy
        kwargs_lens = [self._candidate_kwargs(index, z_s, H0)]
        t_arr = lm.arrival_time(x_img, y_img, kwargs_lens, x_source=x_src, y_source=y_src)
        return t_arr - t_arr.min()

    # ------------------------------------------------------------------
    # χ² family – unifies all previous variants into one fast routine ----
    # ------------------------------------------------------------------
    def chi_squared(
        self,
        params: Sequence[float],
        dt_true: Sequence[float],
        *,
        index: int = 0,
        sigma: float = 0.05,
        sigma_lum: Optional[float] = None,
        lum_dist_true: Optional[float] = None,
    ) -> float:
        """General-purpose χ².

        Parameters
        ----------
        params : (x_src, y_src, z_s [, H0])
        """
        # unpack -------------------------------------------------------
        x_src, y_src, *rest = params
        z_s = rest[0] if rest else self.data.z_source_default
        H0 = rest[1] if len(rest) > 1 else self.data.H0_default

        # images -------------------------------------------------------
        x_img, y_img = self.image_positions(x_src, y_src, z_s, index=index, H0=H0)
        if len(x_img) == 0:
            return 1e4  # no images – penalize hard
        if len(x_img) != len(dt_true):
            return (abs(len(x_img) - len(dt_true))) ** 0.5 * 0.7e4

        dt_model = self.time_delays(x_img, y_img, x_src, y_src, z_s, index=index, H0=H0)
        mask = np.array(dt_true) != 0
        sigma_arr = sigma * np.array(dt_true)
        chi_sq_dt = np.sum(((dt_model[mask] - dt_true[mask]) / sigma_arr[mask]) ** 2)

        # optional luminosity term ------------------------------------
        if sigma_lum is not None and lum_dist_true is not None:
            cosmo = self.cosmology(H0)
            lum_dist_model = cosmo.luminosity_distance(z_s).value
            chi_sq_lum = ((lum_dist_model - lum_dist_true) / (sigma_lum * lum_dist_true)) ** 2
            return chi_sq_dt + chi_sq_lum
        return chi_sq_dt

    # ------------------------------------------------------------------
    # Differential Evolution wrapper -----------------------------------
    # ------------------------------------------------------------------
    def de_search(
        self,
        dt_true: Sequence[float],
        *,
        index: int = 0,
        bounds_xy: float = 50.0,
        bounds_z: Tuple[float, float] | None = None,
        bounds_H0: Tuple[float, float] | None = None,
        maxiter: int = 200,
    ):  # -> Tuple[np.ndarray, float]
        """Run differential evolution to find best parameters."""
        x_c = self.data.x_center[index]
        y_c = self.data.y_center[index]
        bounds = [(x_c - bounds_xy, x_c + bounds_xy), (y_c - bounds_xy, y_c + bounds_xy)]
        if bounds_z is not None:
            bounds.append(bounds_z)
        if bounds_H0 is not None:
            bounds.append(bounds_H0)

        def obj(p):
            return self.chi_squared(p, dt_true, index=index)

        res = differential_evolution(obj, bounds, popsize=40, maxiter=maxiter, polish=False, updating="deferred", workers=-1)
        return res.x, res.fun

    # ------------------------------------------------------------------
    # MCMC wrapper ------------------------------------------------------
    # ------------------------------------------------------------------
    def mcmc_sampler(
        self,
        dt_true: Sequence[float],
        initial: np.ndarray,
        *,
        index: int = 0,
        n_walkers: int = 32,
        n_steps: int = 1000,
        burn_in: int = 300,
    ) -> Tuple[emcee.EnsembleSampler, np.ndarray]:
        """Run MCMC sampling with emcee."""
        ndim = len(initial)
        pos0 = initial + 1e-4 * np.random.randn(n_walkers, ndim)
        sampler = emcee.EnsembleSampler(
            n_walkers,
            ndim,
            lambda p: -0.5 * self.chi_squared(p, dt_true, index=index),
            moves=emcee.moves.StretchMove(a=1.9),
        )
        sampler.run_mcmc(pos0, n_steps, progress=True)
        return sampler, sampler.get_chain(discard=burn_in, flat=True)

    # ------------------------------------------------------------------
    # High‑level pipeline: DE → MCMC -----------------------------------
    # ------------------------------------------------------------------
    def de_then_mcmc(
        self,
        dt_true: Sequence[float],
        *,
        index: int = 0,
        de_kwargs: Optional[Dict] = None,
        mcmc_kwargs: Optional[Dict] = None,
    ):
        """Run DE to find best params, then MCMC to sample around them."""
        de_kwargs = de_kwargs or {}
        mcmc_kwargs = mcmc_kwargs or {}

        best_params, best_chi = self.de_search(dt_true, index=index, **de_kwargs)
        sampler, flat = self.mcmc_sampler(dt_true, best_params, index=index, **mcmc_kwargs)
        med = np.median(flat, axis=0)
        return {
            "de_best": best_params,
            "de_chi": best_chi,
            "mcmc_median": med,
            "sampler": sampler,
            "samples": flat,
        }
