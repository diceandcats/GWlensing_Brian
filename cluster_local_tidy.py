# pylint: skip-file
import os
import sys

# Keep BLAS single-threaded to avoid oversubscription across processes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Reduce glibc arena bloat per process (helps multi-proc memory)
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from collections import OrderedDict

from scipy.optimize import differential_evolution
from scipy.interpolate import RegularGridInterpolator
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import emcee

from lensing_data_class import LensingData

import multiprocessing as mp
from sys import platform


# Global context of DE so workers don't need to pickle 'self'
_DE_SELF = None
_DE_BOUNDS_KEYS = None
_DE_DT_TRUE = None
_DE_INDEX = None
_DE_SIGMA_LUM = None
_DE_LUM_DIST_TRUE = None

def _set_de_context(self_obj, bounds_keys, dt_true, index, sigma_lum, lum_dist_true):
    """Set module-level globals for DE workers (with Linux 'fork', this is inherited by children)."""
    global _DE_SELF, _DE_BOUNDS_KEYS, _DE_DT_TRUE, _DE_INDEX, _DE_SIGMA_LUM, _DE_LUM_DIST_TRUE
    _DE_SELF = self_obj
    _DE_BOUNDS_KEYS = list(bounds_keys)
    _DE_DT_TRUE = np.asarray(dt_true)
    _DE_INDEX = int(index)
    _DE_SIGMA_LUM = sigma_lum
    _DE_LUM_DIST_TRUE = lum_dist_true

def _de_objective_func_global(params: np.ndarray) -> float:
    """Top-level objective (picklable) that reads globals set by _set_de_context."""
    param_map = {key: val for key, val in zip(_DE_BOUNDS_KEYS, params)}
    return _DE_SELF._calculate_chi_squared(
        param_map, _DE_DT_TRUE, _DE_INDEX, sigma_lum=_DE_SIGMA_LUM, lum_dist_true=_DE_LUM_DIST_TRUE
    )


# Global context of MCMC so workers don't need to pickle 'self'
_MCMC_SELF = None
_MCMC_KEYS = None
_MCMC_BOUNDS = None
_MCMC_DT_TRUE = None
_MCMC_INDEX = None
_MCMC_SIGMA_LUM = None
_MCMC_LUM_DIST_TRUE = None

def _set_mcmc_context(self_obj, dt_true, index, bounds_dict, sigma_lum, lum_dist_true):
    global _MCMC_SELF, _MCMC_KEYS, _MCMC_BOUNDS, _MCMC_DT_TRUE, _MCMC_INDEX, _MCMC_SIGMA_LUM, _MCMC_LUM_DIST_TRUE
    _MCMC_SELF = self_obj
    _MCMC_KEYS = list(bounds_dict.keys())
    _MCMC_BOUNDS = np.asarray([bounds_dict[k] for k in _MCMC_KEYS], dtype=float)
    _MCMC_DT_TRUE = np.asarray(dt_true)
    _MCMC_INDEX = int(index)
    _MCMC_SIGMA_LUM = sigma_lum
    _MCMC_LUM_DIST_TRUE = lum_dist_true

def _log_posterior_global(theta: np.ndarray) -> float:
    """Top-level, picklable log-posterior used by emcee workers."""
    for i, (lo, hi) in enumerate(_MCMC_BOUNDS):
        ti = theta[i]
        if (ti < lo) or (ti > hi):
            return -np.inf
    params = {k: v for k, v in zip(_MCMC_KEYS, theta)}
    chi_sq = _MCMC_SELF._calculate_chi_squared(
        params=params,
        dt_true=_MCMC_DT_TRUE,
        index=_MCMC_INDEX,
        sigma_lum=_MCMC_SIGMA_LUM,
        lum_dist_true=_MCMC_LUM_DIST_TRUE,
    )
    return -0.5 * chi_sq


class ClusterLensingUtils:
    """
    Handles data initialization, cosmological calculations, and lensing map scaling.
    Process-safe: heavy, mutable objects are kept per-process and bounded via LRU caches.
    """
    def __init__(self, data: LensingData, z_s_ref: float, cosmo_H0: float = 70.0, cosmo_Om0: float = 0.3):
        self.data = data
        self.z_s_ref = z_s_ref

        # Keep references (no copies) to the large maps; children will share via COW after fork.
        self.alpha_maps_x_orig = data.alpha_maps_x
        self.alpha_maps_y_orig = data.alpha_maps_y
        self.lens_potential_maps_orig = data.lens_potential_maps

        self.map_sizes = [len(m) for m in data.alpha_maps_x]
        self.base_cosmo = FlatLambdaCDM(H0=cosmo_H0, Om0=cosmo_Om0)

        # Precompute grids once (arcsec)
        self._x_grids: List[np.ndarray] = []
        for i in range(len(data.z_l_list)):
            size = self.map_sizes[i]
            pix = np.float32(data.pixscale[i])
            x_grid = np.linspace(0, size - 1, size, dtype=np.float32) * pix
            self._x_grids.append(x_grid)

        # Per-process bounded caches (LRU)
        self._buffers: List[Optional[Dict[str, np.ndarray]]] = [None] * len(self.data.z_l_list)
        self._lm_cache_max = int(os.environ.get("LM_CACHE_MAX_PROC", "8"))
        self._lm_cache: List[OrderedDict] = [OrderedDict() for _ in range(len(self.data.z_l_list))]
        self._cosmo_cache_max = int(os.environ.get("COSMO_CACHE_MAX_PROC", "32"))
        self._cosmo_cache: OrderedDict = OrderedDict()
        self._dist_cache_max = int(os.environ.get("DIST_CACHE_MAX_PROC", "1024"))
        self._dist_cache: OrderedDict = OrderedDict()

        # Prebuild read-only interpolators for sigma_dt (safe to share via COW; inputs are read-only)
        self._sigma_rgi = []
        for sigma in self.data.uncertainty_dt:
            _, n_y = sigma.shape  # deleted unused n_x
            points = (np.arange(n_y), np.arange(sigma.shape[0]))  # (y, x) order
            # Align with earlier code: points = (np.arange(n_y), np.arange(n_x))
            # We keep the original logic: (y, x) indexing with xi stacking below
            points = (np.arange(n_y), np.arange(sigma.shape[0]))
            self._sigma_rgi.append(
                RegularGridInterpolator(points, sigma, method="linear", bounds_error=True)
            )

    def _get_buffers(self, index: int, shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Per-process scratch buffers reused across objective calls."""
        if self._buffers[index] is None:
            self._buffers[index] = {
                "f_": np.empty(shape, dtype=np.float32),
                "f_x": np.empty(shape, dtype=np.float32),
                "f_y": np.empty(shape, dtype=np.float32),
            }
        return self._buffers[index]

    def _get_cosmo(self, H0: float) -> FlatLambdaCDM:
        """
        Get a cosmology model for a specific H0 value, using a cache to avoid
        recreating models unnecessarily.
        """
        key = round(float(H0), 2)
        cosmo = self._cosmo_cache.get(key)
        if cosmo is None:
            cosmo = FlatLambdaCDM(H0=key, Om0=self.base_cosmo.Om0)
            self._cosmo_cache[key] = cosmo
            if len(self._cosmo_cache) > self._cosmo_cache_max:
                self._cosmo_cache.popitem(last=False)
        return cosmo

    def _get_distances(self, z_l: float, z_s: float, H0: float) -> Tuple[float, float]:
        """Per-process LRU of (D_S, D_LS) keyed by quantized (z_l, z_s, H0)."""
        key = (round(float(z_l), 3), round(float(z_s), 3), round(float(H0), 2))
        v = self._dist_cache.get(key)
        if v is None:
            cosmo = self._get_cosmo(H0)
            D_S = cosmo.angular_diameter_distance(z_s).value
            D_LS = cosmo.angular_diameter_distance_z1z2(z_l, z_s).value
            v = (D_S, D_LS)
            self._dist_cache[key] = v
            if len(self._dist_cache) > self._dist_cache_max:
                self._dist_cache.popitem(last=False)
        return v

    def _get_lens_and_solver(self, index: int, z_s: float, H0: float) -> Tuple[LensModel, LensEquationSolver]:
        """Per-process LRU of (LensModel, Solver) keyed by quantized (z_s, H0)."""
        cache = self._lm_cache[index]
        key = (round(float(z_s), 3), round(float(H0), 2))
        pair = cache.get(key)
        if pair is None:
            cosmo = self._get_cosmo(H0)
            lm = LensModel(lens_model_list=["INTERPOL"], z_source=key[0],
                           z_lens=self.data.z_l_list[index], cosmo=cosmo)
            solver = LensEquationSolver(lm)
            cache[key] = (lm, solver)
            if len(cache) > self._lm_cache_max:
                cache.popitem(last=False)  # evict LRU
            return lm, solver
        return pair

    def _get_scaled_kwargs(self, z_s: float, index: int, H0: float) -> Optional[Dict[str, Any]]:
        """Compute scaled maps for (z_s, H0) using cached distances; return kwargs for INTERPOL."""
        D_S, D_LS = self._get_distances(self.data.z_l_list[index], z_s, H0)
        if D_S == 0 or not np.isfinite(D_S) or not np.isfinite(D_LS):
            return None

        scale = np.float32(D_LS / D_S)
        x_grid = self._x_grids[index]
        shape = self.alpha_maps_x_orig[index].shape

        buf = self._get_buffers(index, shape)
        np.multiply(self.lens_potential_maps_orig[index], scale, out=buf["f_"])
        np.multiply(self.alpha_maps_x_orig[index], scale, out=buf["f_x"])
        np.multiply(self.alpha_maps_y_orig[index], scale, out=buf["f_y"])

        return {
            "grid_interp_x": x_grid,
            "grid_interp_y": x_grid,
            "f_": buf["f_"],
            "f_x": buf["f_x"],
            "f_y": buf["f_y"],
        }


class ClusterLensing(ClusterLensingUtils):
    def calculate_imgs_delays_magns(self, params: Dict[str, float], cluster_index: int) -> Dict[str, Any]:
        """Calculate image positions, time delays and magnification for given source parameters and cluster index."""
        x_src, y_src = params["x_src"], params["y_src"]
        z_s = params.get("z_s", self.z_s_ref)
        H0 = params.get("H0", self.base_cosmo.H0.value)

        kwargs = self._get_scaled_kwargs(z_s, cluster_index, H0)
        if kwargs is None:
            return {"image_positions": (np.array([]), np.array([])), "time_delays": np.array([])}

        lens_model, solver = self._get_lens_and_solver(cluster_index, z_s, H0)
        x_img, y_img = solver.image_position_from_source(
            x_src, y_src, [kwargs],
            min_distance=self.data.pixscale[cluster_index],
            search_window=self.data.search_window_list[cluster_index],
            x_center=self.data.x_center[cluster_index],
            y_center=self.data.y_center[cluster_index],
        )
        if len(x_img) == 0:
            return {"image_positions": (np.array([]), np.array([])), "time_delays": np.array([])}

        arrival_times = lens_model.arrival_time(x_img, y_img, [kwargs], x_source=x_src, y_source=y_src)
        mu = lens_model.magnification(x_img, y_img, [kwargs])
        return {"image_positions": (x_img, y_img), "time_delays": arrival_times - np.min(arrival_times), "magnifications": mu}
    
    def calculate_time_delay_uncertainty(self, img: np.ndarray, index: int) -> np.ndarray:
        # img is [x_img, y_img]; RGI expects (y, x)
        xi = np.vstack((np.asarray(img[1]), np.asarray(img[0]))).T
        vals = self._sigma_rgi[index](xi)  # vectorized lookup
        return np.asarray(vals)

    def _calculate_chi_squared(
        self,
        params: Dict[str, float],
        dt_true: np.ndarray,
        index: int,
        sigma_lum: Optional[float] = None,
        lum_dist_true: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate chi-squared between model-predicted and true time delays (and optionally luminosity distance)."""
        x_src, y_src = params["x_src"], params["y_src"]
        z_s = params.get("z_s", self.z_s_ref)
        H0 = params.get("H0", self.base_cosmo.H0.value)

        kwargs = self._get_scaled_kwargs(z_s, index, H0)
        if kwargs is None:
            return 5e5  # invalid cosmology penalty

        lens_model, solver = self._get_lens_and_solver(index, z_s, H0)
        x_img, y_img = solver.image_position_from_source(
            x_src, y_src, [kwargs],
            min_distance=self.data.pixscale[index],
            search_window=self.data.search_window_list[index],
            x_center=self.data.x_center[index],
            y_center=self.data.y_center[index],
        )

        if len(x_img) != len(dt_true):
            penalty = 3e3
            return (abs(len(x_img) - len(dt_true))) ** 0.5 * penalty

        # Relative time delays
        t = lens_model.arrival_time(x_img, y_img, [kwargs], x_source=x_src, y_source=y_src)
        dt_candidate = t - t.min()

        # Magnifications
        mu = lens_model.magnification(x_img, y_img, [kwargs])


        # Chi-squared for time delays
        mask = np.array(dt_true) != 0
        sigma_dt_pixel = self.calculate_time_delay_uncertainty([x_img, y_img], index)
        sigma_arr = sigma_dt_pixel * np.array(dt_true)
        chi_sq_dt = np.sum((dt_candidate[mask] - dt_true[mask]) ** 2 / sigma_arr[mask] ** 2)
        
        chi_sq_lum = 0.0
        # Chi-squared for luminosity distance (if provided)
        if sigma_lum is not None and lum_dist_true is not None:
            cosmo = self._get_cosmo(H0)
            lum_dist_unlensed = cosmo.luminosity_distance(z_s).value
            lum_dist_candidate = [lum_dist_unlensed / np.abs(m) for m in mu]

            chi_sq_lum = np.sum((lum_dist_candidate - lum_dist_true) ** 2 / (sigma_lum * lum_dist_true) ** 2)

        return float(chi_sq_dt + chi_sq_lum)

    def run_de_optimization(
        self,
        dt_true: np.ndarray,
        index: int,
        fit_z: bool = False,
        fit_hubble: bool = False,
        lum_dist_true: Optional[float] = None,
        sigma_lum: Optional[float] = None,
        de_settings: Dict[str, Any] = None,
    ) -> Tuple[Dict, float]:
        """Run Differential Evolution to find best-fit parameters for a given cluster index."""
        bounds = {
            "x_src": (self.data.x_center[index] - 50, self.data.x_center[index] + 50),
            "y_src": (self.data.y_center[index] - 50, self.data.y_center[index] + 50),
        }
        if fit_z:
            bounds["z_s"] = (1.0, 5.0)
        if fit_hubble:
            bounds["H0"] = (60, 80)

        _set_de_context(self, list(bounds.keys()), dt_true, index, sigma_lum, lum_dist_true)

        default_settings = {
            "strategy": "rand1bin",
            "maxiter": 200,
            "popsize": int(os.environ.get("DE_POPSIZE", "50")),
            "tol": 1e-7,
            "mutation": (0.5, 1),
            "recombination": 0.7,
            "polish": bool(int(os.environ.get("DE_POLISH", "0"))),
            "updating": "deferred",
            "disp": True,
            "init": os.environ.get("DE_INIT", "latinhypercube"),
        }
        if de_settings:
            default_settings.update(de_settings)

        estop = default_settings.pop("early_stop_threshold", 0.03)
        if estop is not None:
            def _cb(xk, conv):
                return _de_objective_func_global(xk) < estop
            default_settings["callback"] = _cb

        try:
            slot_cpus = len(os.sched_getaffinity(0))
        except Exception:
            slot_cpus = os.cpu_count() or 1

        env_override = os.environ.get("DE_PROCS")
        if env_override:
            try:
                n_procs = max(1, min(int(env_override), slot_cpus))
            except ValueError:
                n_procs = slot_cpus
        else:
            n_procs = slot_cpus

        ctx = mp.get_context("fork") if platform.startswith("linux") else mp.get_context()
        print(
            f"[DE] Process Pool with {n_procs} procs (init={default_settings.get('init')}, popsize={default_settings.get('popsize')})",
            flush=True,
        )

        with ctx.Pool(processes=n_procs) as pool:
            result = differential_evolution(
                _de_objective_func_global,
                list(bounds.values()),
                workers=pool.map,
                **default_settings,
            )

        best_params = {k: v for k, v in zip(bounds.keys(), result.x)}
        return best_params, result.fun

    def run_mcmc_sampler(
        self,
        dt_true: np.ndarray,
        index: int,
        initial_params: Dict[str, float],
        mcmc_settings: Dict[str, Any],
    ) -> emcee.EnsembleSampler:
        """Run MCMC sampling to explore parameter space around initial_params for a given cluster index."""
        n_walkers = mcmc_settings.get("n_walkers", 32)
        n_steps = mcmc_settings.get("n_steps", 1000)
        sigma_lum = mcmc_settings.get("sigma_lum", 0.05)
        lum_dist_true = mcmc_settings.get("lum_dist_true")

        bounds = {
            "x_src": mcmc_settings.get("x_bounds", (initial_params["x_src"] - 5, initial_params["x_src"] + 5)),
            "y_src": mcmc_settings.get("y_bounds", (initial_params["y_src"] - 5, initial_params["y_src"] + 5)),
        }
        if "z_s" in initial_params:
            bounds["z_s"] = mcmc_settings.get("z_bounds", (1.0, 5.0))
        if "H0" in initial_params:
            bounds["H0"] = mcmc_settings.get("H0_bounds", (53, 92))

        ndim = len(bounds)
        initial_state = np.array([initial_params[k] for k in bounds.keys()], dtype=float)

        for j, k in enumerate(bounds.keys()):
            lo, hi = bounds[k]
            if not (lo <= initial_state[j] <= hi):
                raise ValueError(f"Initial parameter '{k}'={initial_state[j]:.3f} outside prior {bounds[k]}.")

        p0 = np.empty((n_walkers, ndim), dtype=float)
        rng = np.random.default_rng()
        for i in range(n_walkers):
            while True:
                prop = initial_state + 1e-4 * rng.standard_normal(ndim)
                if all(bounds[k][0] <= prop[j] <= bounds[k][1] for j, k in enumerate(bounds.keys())):
                    p0[i] = prop
                    break

        _set_mcmc_context(self, dt_true, index, bounds, sigma_lum, lum_dist_true)

        try:
            slot_cpus = len(os.sched_getaffinity(0))
        except Exception:
            slot_cpus = os.cpu_count() or 1

        env_override = os.environ.get("MCMC_PROCS")
        if env_override:
            try:
                n_procs = max(1, min(int(env_override), slot_cpus))
            except ValueError:
                n_procs = slot_cpus
        else:
            n_procs = slot_cpus

        ctx = mp.get_context("fork") if platform.startswith("linux") else mp.get_context()
        print(f"[MCMC] Process Pool with {n_procs} procs", flush=True)
        with ctx.Pool(processes=n_procs) as pool:
            sampler = emcee.EnsembleSampler(n_walkers, ndim, _log_posterior_global, pool=pool)

            # tqdm -> STDOUT; update roughly once every 100s
            progress_kwargs = {"file": sys.stdout, "mininterval": 100.0}
            sampler.run_mcmc(p0, n_steps, progress=True, progress_kwargs=progress_kwargs)

        return sampler

    def find_best_fit(
        self,
        dt_true: np.ndarray,
        run_mcmc: bool = False,
        de_settings: Optional[Dict] = None,
        mcmc_settings: Optional[Dict] = None,
    ) -> Tuple[List[Dict], List[int]]:
        """Run DE optimization across all clusters, optionally followed by MCMC sampling."""
        results = []
        mcmc_settings = mcmc_settings or {}
        # Variables required for DE follow mcmc_settings, de_settings are for population, mutation, etc.
        fit_z = mcmc_settings.get("fit_z", False)
        fit_hubble = mcmc_settings.get("fit_hubble", False)
        lum_dist_true = mcmc_settings.get("lum_dist_true")
        sigma_lum = mcmc_settings.get("sigma_lum")

        for i in range(len(self.data.z_l_list)):
            print(f"--- Running DE for Cluster {i} ---", flush=True)
            best_params, min_chi_sq = self.run_de_optimization(
                dt_true,
                i,
                fit_z=fit_z,
                fit_hubble=fit_hubble,
                lum_dist_true=lum_dist_true,
                sigma_lum=sigma_lum,
                de_settings=de_settings,
            )
            results.append({"cluster_index": i, "de_params": best_params, "chi_sq": min_chi_sq})
            print(f"Cluster {i} DE best chi^2: {min_chi_sq:.3f} with params {best_params}", flush=True)

        best_result = min(results, key=lambda x: x["chi_sq"])
        print("\n--- DE Optimization Summary ---", flush=True)
        print(f"Best fit found for Cluster {best_result['cluster_index']} with chi^2 = {best_result['chi_sq']:.3f}", flush=True)
        print(f"Best-fit parameters: {best_result['de_params']}", flush=True)

        accepted_clusters = []
        if run_mcmc:
            mcmc_threshold = 2.3 + best_result["chi_sq"]
            for result in results:
                if result["chi_sq"] <= mcmc_threshold:
                    accepted_clusters.append(result["cluster_index"])
                    print(f"\n--- Running MCMC for Cluster {result['cluster_index']} (chi_sq={result['chi_sq']:.3f}) ---", flush=True)
                    sampler = self.run_mcmc_sampler(
                        dt_true, result["cluster_index"], result["de_params"], mcmc_settings
                    )
                    result["mcmc_sampler"] = sampler
            results = [res for res in results if "mcmc_sampler" in res]

        return results, accepted_clusters

    def save_mcmc_results(
        self,
        sampler: emcee.EnsembleSampler,
        best_result: Dict,
        n_burn_in: int,
        output_path: str,
        dt_true: np.ndarray,
        mcmc_settings: Dict,
    ):
        """Process and save MCMC results to a compressed .npz file."""
        print(f"Processing results for Cluster {best_result['cluster_index']}...", flush=True)

        flat_samples = sampler.get_chain(discard=n_burn_in, flat=True)
        log_probs = sampler.get_log_prob(discard=n_burn_in, flat=True)
        full_chain = sampler.get_chain(discard=n_burn_in)

        medians_array = np.median(flat_samples, axis=0)
        param_labels = list(best_result["de_params"].keys())
        median_params = {key: val for key, val in zip(param_labels, medians_array)}

        chi_sq_final = self._calculate_chi_squared(
            params=median_params,
            dt_true=dt_true,
            index=best_result["cluster_index"],
            sigma_lum=mcmc_settings.get("sigma_lum"),
            lum_dist_true=mcmc_settings.get("lum_dist_true"),
        )

        np.savez_compressed(
            output_path,
            chain=full_chain,
            flat_chain=flat_samples,
            log_prob=log_probs,
            median_params=medians_array,
            param_labels=param_labels,
            chi_sq_for_median=chi_sq_final,
        )
        print(f"-> Saved {flat_samples.shape[0]} samples to {output_path}", flush=True)

    def generate_source_positions(
        self,
        n_samples: int,
        index: int,
        search_range: float,
        img_no: int,
        H0: Optional[float] = None,
        z_s: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate random source positions around cluster center that yield a specific number of images."""
        x_center = self.data.x_center[index]
        y_center = self.data.y_center[index]

        n = 0
        x_srcs = np.array([])
        y_srcs = np.array([])
        z_ss = np.array([])
        H0s = np.array([])

        H0_arg = H0
        z_s_arg = z_s

        while n < n_samples:
            x_src = np.random.uniform(x_center - search_range, x_center + search_range)
            y_src = np.random.uniform(y_center - search_range, y_center + search_range)

            current_z_s = z_s_arg if z_s_arg is not None else np.random.uniform(1.5, 4.0)
            current_H0 = H0_arg if H0_arg is not None else np.random.uniform(65, 76)

            test_params = {"x_src": x_src, "y_src": y_src, "z_s": current_z_s, "H0": current_H0}
            test_cluster = index
            output = self.calculate_imgs_delays_magns(test_params, test_cluster)
            if len(output["image_positions"][0]) == img_no:
                n += 1
                x_srcs = np.append(x_srcs, x_src)
                y_srcs = np.append(y_srcs, y_src)
                z_ss = np.append(z_ss, current_z_s)
                H0s = np.append(H0s, current_H0)
                print(x_src, y_src, current_z_s, current_H0, flush=True)

        return index, x_srcs, y_srcs, z_ss, H0s
