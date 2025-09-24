# pylint: skip-file
# ========================= cluster_local_tidy.py (with MOD 1, MOD 2, and pickling FIX) =========================
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import differential_evolution
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import emcee
from typing import List, Optional, Tuple, Dict, Any

from lensing_data_class import LensingData
from scipy.interpolate import interpn

# ---------------------------------------------------------------------------------------------
# [MOD 1] process-based parallelism imports (no threads for DE)
# ---------------------------------------------------------------------------------------------
import multiprocessing as mp
from sys import platform

import threading


# =============================================================================================
# [FIX] GLOBAL CONTEXT FOR DE to avoid pickling 'self' (which holds a threading.local)
# =============================================================================================
_DE_SELF = None
_DE_BOUNDS_KEYS = None
_DE_DT_TRUE = None
_DE_INDEX = None
_DE_SIGMA_LUM = None
_DE_LUM_DIST_TRUE = None

def _set_de_context(self_obj, bounds_keys, dt_true, index, sigma_lum, lum_dist_true):
    """Set module-level globals so workers don't need to pickle 'self'."""
    global _DE_SELF, _DE_BOUNDS_KEYS, _DE_DT_TRUE, _DE_INDEX, _DE_SIGMA_LUM, _DE_LUM_DIST_TRUE
    _DE_SELF = self_obj
    _DE_BOUNDS_KEYS = list(bounds_keys)
    _DE_DT_TRUE = np.asarray(dt_true)
    _DE_INDEX = int(index)
    _DE_SIGMA_LUM = sigma_lum
    _DE_LUM_DIST_TRUE = lum_dist_true

# [FIX] Top-level, no-extra-args objective used by DE workers (picklable).
def _de_objective_func_global(params: np.ndarray) -> float:
    param_map = {key: val for key, val in zip(_DE_BOUNDS_KEYS, params)}
    return _DE_SELF._calculate_chi_squared(
        param_map, _DE_DT_TRUE, _DE_INDEX, sigma_lum=_DE_SIGMA_LUM, lum_dist_true=_DE_LUM_DIST_TRUE
    )

# ===== MCMC global context (NEW) =====
_MCMC_SELF = None
_MCMC_KEYS = None          # e.g. ["x_src","y_src","z_s","H0"]
_MCMC_BOUNDS = None        # np.array([[lo,hi],...])
_MCMC_DT_TRUE = None
_MCMC_INDEX = None
_MCMC_SIGMA_LUM = None
_MCMC_LUM_DIST_TRUE = None

def _set_mcmc_context(self_obj, dt_true, index, bounds_dict, sigma_lum, lum_dist_true):
    """Populate globals so the pool workers don't need to pickle 'self'."""
    global _MCMC_SELF, _MCMC_KEYS, _MCMC_BOUNDS, _MCMC_DT_TRUE, _MCMC_INDEX, _MCMC_SIGMA_LUM, _MCMC_LUM_DIST_TRUE
    _MCMC_SELF = self_obj
    _MCMC_KEYS = list(bounds_dict.keys())  # preserves insertion order
    _MCMC_BOUNDS = np.asarray([bounds_dict[k] for k in _MCMC_KEYS], dtype=float)
    _MCMC_DT_TRUE = np.asarray(dt_true)
    _MCMC_INDEX = int(index)
    _MCMC_SIGMA_LUM = sigma_lum
    _MCMC_LUM_DIST_TRUE = lum_dist_true

def _log_posterior_global(theta: np.ndarray) -> float:
    """Top-level, picklable log-prob using the global MCMC context."""
    # uniform priors
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
        lum_dist_true=_MCMC_LUM_DIST_TRUE
    )
    return -0.5 * chi_sq

# --- Global Function for MCMC Parallelization (unchanged) ---

# def _log_posterior_func(params: np.ndarray,
#                         self_obj: 'ClusterLensing',
#                         dt_true: np.ndarray,
#                         index: int,
#                         bounds: Dict[str, Tuple[float, float]],
#                         sigma_lum: Optional[float],
#                         lum_dist_true: Optional[float]) -> float:
#     param_map = {"x_src": params[0], "y_src": params[1]}
#     fit_z = "z_s" in bounds
#     fit_hubble = "H0" in bounds

#     if fit_z:
#         param_map["z_s"] = params[2]
#         if fit_hubble:
#             param_map["H0"] = params[3]
#     elif fit_hubble:
#         param_map["H0"] = params[2]

#     for i, key in enumerate(bounds.keys()):
#         if not (bounds[key][0] <= params[i] <= bounds[key][1]):
#             return -np.inf

#     chi_sq = self_obj._calculate_chi_squared(
#         param_map, dt_true, index, sigma_lum, lum_dist_true
#     )
#     return -0.5 * chi_sq


# --- Utility Class ---

class ClusterLensingUtils:
    """
    Handles data initialization, cosmological calculations, and lensing map scaling.
    """
    def __init__(self, data: LensingData, z_s_ref: float, cosmo_H0: float = 70.0, cosmo_Om0: float = 0.3):
        self.data = data
        self.z_s_ref = z_s_ref
        
        # keep references instead of making copies (save RAM)
        self.alpha_maps_x_orig = data.alpha_maps_x
        self.alpha_maps_y_orig = data.alpha_maps_y
        self.lens_potential_maps_orig = data.lens_potential_maps
        
        self.map_sizes = [len(m) for m in data.alpha_maps_x]
        
        self._cosmo_cache = {}
        self.base_cosmo = self._get_cosmology(cosmo_H0, cosmo_Om0)

        # Precompute grids once
        self._x_grids: List[np.ndarray] = []
        for i in range(len(data.z_l_list)):
            size = self.map_sizes[i]
            pix = np.float32(data.pixscale[i])
            x_grid = np.linspace(0, size - 1, size, dtype=np.float32) * pix
            self._x_grids.append(x_grid)

        # per-thread buffers (harmless with processes too)
        self._tls = threading.local()

        # -------------------------------------------------------------------------------------
        # [MOD 2] Cache one LensModel + one Solver PER CLUSTER and REUSE them
        # -------------------------------------------------------------------------------------
        self.ref_lens_models = []
        self.ref_solvers = []
        for i in range(len(data.z_l_list)):
            lens_model = LensModel(
                lens_model_list=['INTERPOL'],
                z_source=z_s_ref,                   # kwargs will carry the redshift scaling
                z_lens=data.z_l_list[i],
                cosmo=self.base_cosmo
            )
            self.ref_lens_models.append(lens_model)
            self.ref_solvers.append(LensEquationSolver(lens_model))

    def _get_thread_buffers(self, index: int, shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        if not hasattr(self._tls, "buffers"):
            self._tls.buffers = [None] * len(self.data.z_l_list)
        if self._tls.buffers[index] is None:
            self._tls.buffers[index] = {
                'f_' : np.empty(shape, dtype=np.float32),
                'f_x': np.empty(shape, dtype=np.float32),
                'f_y': np.empty(shape, dtype=np.float32),
            }
        return self._tls.buffers[index]

    def _get_cosmology(self, H0: float, Om0: float = 0.3) -> FlatLambdaCDM:
        if H0 not in self._cosmo_cache:
            self._cosmo_cache[H0] = FlatLambdaCDM(H0=H0, Om0=Om0)
        return self._cosmo_cache[H0]

    def _get_scaled_model_and_kwargs(self, z_s: float, index: int, H0: float) -> Tuple[Optional[LensModel], Optional[Dict[str, Any]]]:
        """
        Compute the scaled deflection/potential maps for (z_s, H0) and
        return the REUSED per-cluster LensModel + kwargs. Returns (None, None) if invalid.
        """
        # keep per-H0 distances here to preserve original physics
        cosmo = self._get_cosmology(H0)
        D_S = cosmo.angular_diameter_distance(z_s).value
        D_LS = cosmo.angular_diameter_distance_z1z2(self.data.z_l_list[index], z_s).value
        if D_S == 0 or not np.isfinite(D_S) or not np.isfinite(D_LS):
            return None, None

        scale = np.float32(D_LS / D_S)
        x_grid = self._x_grids[index]
        shape = self.alpha_maps_x_orig[index].shape

        buf = self._get_thread_buffers(index, shape)
        np.multiply(self.lens_potential_maps_orig[index], scale, out=buf['f_'])
        np.multiply(self.alpha_maps_x_orig[index],        scale, out=buf['f_x'])
        np.multiply(self.alpha_maps_y_orig[index],        scale, out=buf['f_y'])
        
        kwargs = {
            'grid_interp_x': x_grid,
            'grid_interp_y': x_grid,
            'f_': buf['f_'],
            'f_x': buf['f_x'],
            'f_y': buf['f_y']
        }

        # [MOD 2] REUSE cached model
        lens_model = self.ref_lens_models[index]
        return lens_model, kwargs


# --- Main Analysis Class ---

class ClusterLensing(ClusterLensingUtils):
    def calculate_images_and_delays(self, params: Dict[str, float], cluster_index: int) -> Dict[str, Any]:
        x_src, y_src = params["x_src"], params["y_src"]
        z_s = params.get("z_s", self.z_s_ref)
        H0 = params.get("H0", self.base_cosmo.H0.value)
        
        lens_model, kwargs = self._get_scaled_model_and_kwargs(z_s, cluster_index, H0)
        if lens_model is None:
            print("Warning: Invalid cosmological parameters (e.g., z_s <= z_l).")
            return {'image_positions': (np.array([]), np.array([])), 'time_delays': np.array([])}

        # [MOD 2] REUSE cached solver
        solver = self.ref_solvers[cluster_index]
        x_img, y_img = solver.image_position_from_source(
            x_src, y_src, [kwargs],
            min_distance=self.data.pixscale[cluster_index],
            search_window=self.data.search_window_list[cluster_index],
            x_center=self.data.x_center[cluster_index],
            y_center=self.data.y_center[cluster_index]
        )
        
        if len(x_img) == 0:
            time_delays = np.array([])
        else:
            arrival_times = lens_model.arrival_time(x_img, y_img, [kwargs], x_source=x_src, y_source=y_src)
            time_delays = arrival_times - np.min(arrival_times)
        
        return {'image_positions': (x_img, y_img), 'time_delays': np.sort(time_delays)}

    def calculate_time_delay_uncertainty(self, img: np.ndarray, index: int) -> np.ndarray:
        sigma_dt = self.data.uncertainty_dt[index]
        map_size = sigma_dt.shape[0]
        points = (np.arange(map_size), np.arange(map_size))
        xi = np.vstack((img[1], img[0])).T
        sigma_dt_values = interpn(
            points, 
            sigma_dt, 
            xi,
            method='linear', 
            bounds_error=True
        )
        return np.array(sigma_dt_values)

    def _calculate_chi_squared(self,
                               params: Dict[str, float],
                               dt_true: np.ndarray,
                               index: int,
                               sigma_lum: Optional[float] = None,
                               lum_dist_true: Optional[float] = None) -> float:
        x_src, y_src = params["x_src"], params["y_src"]
        z_s = params.get("z_s", self.z_s_ref)
        H0 = params.get("H0", self.base_cosmo.H0.value)
        
        lens_model, kwargs = self._get_scaled_model_and_kwargs(z_s, index, H0)
        if lens_model is None:
            return 5e5  # Penalty for invalid cosmology

        # [MOD 2] REUSE cached solver
        solver = self.ref_solvers[index]
        x_img, y_img = solver.image_position_from_source(
            x_src, y_src, [kwargs],
            min_distance=self.data.pixscale[index],
            search_window=self.data.search_window_list[index],
            x_center=self.data.x_center[index],
            y_center=self.data.y_center[index]
        )
        
        if len(x_img) != len(dt_true):
            penalty = 3e3
            return (abs(len(x_img) - len(dt_true)))**0.5 * penalty

        t = lens_model.arrival_time(x_img, y_img, [kwargs], x_source=x_src, y_source=y_src)
        dt_candidate = t - t.min()
        
        mask = np.array(dt_true) != 0
        sigma_dt_pixel = self.calculate_time_delay_uncertainty([x_img, y_img], index)
        sigma_arr = sigma_dt_pixel * np.array(dt_true)
        chi_sq_dt = np.sum((dt_candidate[mask] - dt_true[mask])**2 / sigma_arr[mask]**2)
        
        chi_sq_lum = 0.0
        if "H0" in params and lum_dist_true is not None and sigma_lum is not None:
            cosmo = self._get_cosmology(H0)
            lum_dist_candidate = cosmo.luminosity_distance(z_s).value
            chi_sq_lum = (lum_dist_candidate - lum_dist_true)**2 / (sigma_lum * lum_dist_true)**2
            
        return chi_sq_dt + chi_sq_lum

    def run_de_optimization(self,
                            dt_true: np.ndarray,
                            index: int,
                            fit_z: bool = False,
                            fit_hubble: bool = False,
                            lum_dist_true: Optional[float] = None,
                            sigma_lum: Optional[float] = None,
                            de_settings: Dict[str, Any] = None) -> Tuple[Dict, float]:
        """
        Runs differential evolution to find the best-fit parameters.
        """
        bounds = {
            "x_src": (self.data.x_center[index] - 50, self.data.x_center[index] + 50),
            "y_src": (self.data.y_center[index] - 50, self.data.y_center[index] + 50)
        }
        if fit_z:
            bounds["z_s"] = (1.0, 5.0)
        if fit_hubble:
            bounds["H0"] = (60, 80)

        # ---------------------------------------------------------------------------------
        # [FIX] Set global DE context so workers don't receive (unpicklable) 'self' via args
        # ---------------------------------------------------------------------------------
        _set_de_context(self, list(bounds.keys()), dt_true, index, sigma_lum, lum_dist_true)

        default_settings = {
            'strategy': 'rand1bin', 'maxiter': 200, 'popsize': 50, 'tol': 1e-7,
            'mutation': (0.5, 1), 'recombination': 0.7, 'polish': False,
            'updating': 'deferred',
            'disp': True
        }
        if de_settings:
            default_settings.update(de_settings)

        early_stop_threshold = default_settings.pop('early_stop_threshold', 0.05)
        if early_stop_threshold is not None:
            # [FIX] callback uses the global objective (no args)
            def callback_fn(xk, convergence):
                return _de_objective_func_global(xk) < early_stop_threshold
            default_settings['callback'] = callback_fn

        # ---------------------------------------------------------------------------------
        # [MOD 1] Use a PROCESS pool (Linux 'fork' keeps big arrays shared COW)
        # ---------------------------------------------------------------------------------

        # Choose number of worker processes in a Condor-friendly way
        try:
            # Respect cgroup/affinity set by the slot
            n_available = len(os.sched_getaffinity(0))
        except Exception:
            n_available = os.cpu_count() or 1

        # Allow override via environment (set in condor submit)
        env_override = os.environ.get('DE_PROCS')  # use 'MCMC_PROCS' in run_mcmc_sampler
        if env_override:
            try:
                n_procs = max(1, min(int(env_override), n_available))
            except ValueError:
                n_procs = n_available
        else:
            n_procs = n_available
        ctx = mp.get_context('fork') if platform.startswith('linux') else mp.get_context()

        with ctx.Pool(processes=n_procs) as pool:
            result = differential_evolution(
                _de_objective_func_global,          # [FIX] no extra args
                list(bounds.values()),
                workers=pool.map,                   # map(func, iterable)
                **default_settings
            )
        
        best_params = {key: val for key, val in zip(bounds.keys(), result.x)}
        return best_params, result.fun

    def run_mcmc_sampler(self,
                        dt_true: np.ndarray,
                        index: int,
                        initial_params: Dict[str, float],
                        mcmc_settings: Dict[str, Any]) -> emcee.EnsembleSampler:
        n_walkers = mcmc_settings.get("n_walkers", 32)
        n_steps   = mcmc_settings.get("n_steps", 1000)
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
        initial_state = np.array([initial_params[k] for k in bounds.keys()])

        # sanity: start inside priors
        for j, k in enumerate(bounds.keys()):
            lo, hi = bounds[k]
            if not (lo <= initial_state[j] <= hi):
                raise ValueError(f"Initial parameter '{k}'={initial_state[j]:.3f} outside prior {bounds[k]}.")

        # initialize walkers tightly around the initial state
        p0 = np.zeros((n_walkers, ndim))
        for i in range(n_walkers):
            while True:
                prop = initial_state + 1e-4 * np.random.randn(ndim)
                if all(bounds[k][0] <= prop[j] <= bounds[k][1] for j, k in enumerate(bounds.keys())):
                    p0[i] = prop
                    break

        # ===== NEW: set global context & run with a process pool =====
        _set_mcmc_context(self, dt_true, index, bounds, sigma_lum, lum_dist_true)

        # Choose number of worker processes in a Condor-friendly way
        try:
            # Respect cgroup/affinity set by the slot
            n_available = len(os.sched_getaffinity(0))
        except Exception:
            n_available = os.cpu_count() or 1

        # Allow override via environment (set in condor submit)
        env_override = os.environ.get('DE_PROCS')  # use 'MCMC_PROCS' in run_mcmc_sampler
        if env_override:
            try:
                n_procs = max(1, min(int(env_override), n_available))
            except ValueError:
                n_procs = n_available
        else:
            n_procs = n_available
            
        ctx = mp.get_context('fork') if platform.startswith('linux') else mp.get_context()

        with ctx.Pool(processes=n_procs) as pool:
            sampler = emcee.EnsembleSampler(
                n_walkers, ndim, _log_posterior_global, pool=pool
            )
            sampler.run_mcmc(p0, n_steps, progress=True)

        return sampler


    def find_best_fit(self,
                      dt_true: np.ndarray,
                      run_mcmc: bool = False,
                      de_settings: Optional[Dict] = None,
                      mcmc_settings: Optional[Dict] = None) -> Tuple[List[Dict], List[int]]:
        results = []
        mcmc_settings = mcmc_settings or {}
        fit_z = mcmc_settings.get("fit_z", False)
        fit_hubble = mcmc_settings.get("fit_hubble", False)
        lum_dist_true = mcmc_settings.get("lum_dist_true")
        sigma_lum = mcmc_settings.get("sigma_lum")

        for i in range(len(self.data.z_l_list)):
            print(f"--- Running DE for Cluster {i} ---")
            best_params, min_chi_sq = self.run_de_optimization(
                dt_true, i, 
                fit_z=fit_z, 
                fit_hubble=fit_hubble,
                lum_dist_true=lum_dist_true,
                sigma_lum=sigma_lum,
                de_settings=de_settings
            )
            results.append({'cluster_index': i, 'de_params': best_params, 'chi_sq': min_chi_sq})
        
        best_result = min(results, key=lambda x: x['chi_sq'])
        print("\n--- DE Optimization Summary ---")
        print(f"Best fit found for Cluster {best_result['cluster_index']} with chi^2 = {best_result['chi_sq']:.3f}")
        print(f"Best-fit parameters: {best_result['de_params']}")

        accepted_clusters = []
        if run_mcmc:
            mcmc_threshold = 2.3 + best_result['chi_sq']
            for result in results:
                if result['chi_sq'] <= mcmc_threshold:
                    accepted_clusters.append(result['cluster_index'])
                    print(f"\n--- Running MCMC for Cluster {result['cluster_index']} (chi_sq={result['chi_sq']:.3f}) ---")
                    sampler = self.run_mcmc_sampler(
                        dt_true,
                        result['cluster_index'],
                        result['de_params'],
                        mcmc_settings
                    )
                    result['mcmc_sampler'] = sampler
            results = [res for res in results if 'mcmc_sampler' in res]
        
        return results, accepted_clusters
    
    def save_mcmc_results(self,
                          sampler: emcee.EnsembleSampler,
                          best_result: Dict,
                          n_burn_in: int,
                          output_path: str,
                          dt_true: np.ndarray,
                          mcmc_settings: Dict):
        print(f"Processing results for Cluster {best_result['cluster_index']}...")

        flat_samples = sampler.get_chain(discard=n_burn_in, flat=True)
        log_probs = sampler.get_log_prob(discard=n_burn_in, flat=True)
        full_chain = sampler.get_chain(discard=n_burn_in)

        medians_array = np.median(flat_samples, axis=0)
        param_labels = list(best_result['de_params'].keys())
        median_params = {key: val for key, val in zip(param_labels, medians_array)}

        chi_sq_final = self._calculate_chi_squared(
            params=median_params,
            dt_true=dt_true,
            index=best_result['cluster_index'],
            sigma_lum=mcmc_settings.get("sigma_lum"),
            lum_dist_true=mcmc_settings.get("lum_dist_true")
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
        print(f"-> Saved {flat_samples.shape[0]} samples to {output_path}")
# ======================= end cluster_local_tidy.py (MOD 1, MOD 2, and FIX) =======================
