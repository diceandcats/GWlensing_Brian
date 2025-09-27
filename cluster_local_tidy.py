# pylint: skip-file
# ========================= cluster_local_tidy.py =========================
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import numpy as np
from scipy.optimize import differential_evolution
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import emcee
from typing import List, Optional, Tuple, Dict, Any
from lensing_data_class import LensingData
from scipy.interpolate import interpn

import multiprocessing as mp
from sys import platform
import threading
import sys

# ------------------------ DE global context ------------------------
_DE_SELF = None
_DE_BOUNDS_KEYS = None
_DE_DT_TRUE = None
_DE_INDEX = None
_DE_SIGMA_LUM = None
_DE_LUM_DIST_TRUE = None

def _set_de_context(self_obj, bounds_keys, dt_true, index, sigma_lum, lum_dist_true):
    global _DE_SELF, _DE_BOUNDS_KEYS, _DE_DT_TRUE, _DE_INDEX, _DE_SIGMA_LUM, _DE_LUM_DIST_TRUE
    _DE_SELF = self_obj
    _DE_BOUNDS_KEYS = list(bounds_keys)
    _DE_DT_TRUE = np.asarray(dt_true)
    _DE_INDEX = int(index)
    _DE_SIGMA_LUM = sigma_lum
    _DE_LUM_DIST_TRUE = lum_dist_true

_DE_CALLS = 0
_DE_T0 = None
_DE_VERBOSE = bool(int(os.environ.get("DEBUG_DE", "0")))

def _de_objective_func_global(params: np.ndarray) -> float:
    global _DE_CALLS, _DE_T0
    if _DE_T0 is None:
        _DE_T0 = time.time()
    _DE_CALLS += 1
    try:
        param_map = {key: val for key, val in zip(_DE_BOUNDS_KEYS, params)}
        val = _DE_SELF._calculate_chi_squared(
            param_map, _DE_DT_TRUE, _DE_INDEX,
            sigma_lum=_DE_SIGMA_LUM, lum_dist_true=_DE_LUM_DIST_TRUE
        )
        if _DE_VERBOSE and (_DE_CALLS % 100 == 0):
            dt = time.time() - _DE_T0
            print(f"[DE/W] pid={os.getpid()} calls={_DE_CALLS} avg={dt/_DE_CALLS:.4f}s", flush=True)
        return val
    except Exception as e:
        print(f"[DE/W-EXC] pid={os.getpid()} error={type(e).__name__}: {e}", flush=True)
        raise

def _de_pool_initializer(self_obj, bounds_keys, dt_true, index, sigma_lum, lum_dist_true):
    _set_de_context(self_obj, bounds_keys, dt_true, index, sigma_lum, lum_dist_true)
    if os.environ.get("DEBUG_PAR"):
        print(f"[DE/INIT] worker pid={os.getpid()} ready", flush=True)

# ---- top-level probe (picklable) used by DE self-test
def _probe_de(_):
    s = 0
    for k in range(150_000):
        s += (k % 7) * (k % 13)
    return os.getpid(), s

def _parallel_self_test(n_procs, ctx):
    if not os.environ.get("DEBUG_PAR"):
        return
    print(f"[PAR-TEST] launching {n_procs} workers...", flush=True)
    with ctx.Pool(processes=n_procs) as p:
        out = list(p.map(_probe_de, range(n_procs * 2)))
    pids = sorted(set(pid for pid, _ in out))
    print(f"[PAR-TEST] ok: unique_workers={len(pids)} pids={pids} elapsed={0:.2f}s", flush=True)

# ------------------------ MCMC global context ------------------------
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

_MCMC_VERBOSE = bool(int(os.environ.get("DEBUG_MCMC", "0")))
_MCMC_CALLS = 0
_MCMC_T0 = None

def _log_posterior_global(theta: np.ndarray) -> float:
    global _MCMC_CALLS, _MCMC_T0
    if _MCMC_T0 is None:
        _MCMC_T0 = time.time()
    _MCMC_CALLS += 1

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
    if _MCMC_VERBOSE and (_MCMC_CALLS % 500 == 0):
        dt = time.time() - _MCMC_T0
        print(f"[MCMC/W] pid={os.getpid()} calls={_MCMC_CALLS} avg={dt/_MCMC_CALLS:.4f}s", flush=True)
    return -0.5 * chi_sq

def _mcmc_pool_initializer(self_obj, dt_true, index, bounds_dict, sigma_lum, lum_dist_true):
    _set_mcmc_context(self_obj, dt_true, index, bounds_dict, sigma_lum, lum_dist_true)
    if os.environ.get("DEBUG_PAR"):
        print(f"[MCMC/INIT] worker pid={os.getpid()} ready", flush=True)

# ---- top-level probe (picklable) used by MCMC self-test
def _probe_mcmc(_):
    s = 0
    for k in range(120_000):
        s += (k % 7) * (k % 11)
    return os.getpid(), s

def _parallel_self_test_mcmc(n_procs, ctx):
    if not os.environ.get("DEBUG_PAR"):
        return
    print(f"[MCMC-TEST] launching {n_procs} workers...", flush=True)
    with ctx.Pool(processes=n_procs) as p:
        out = list(p.map(_probe_mcmc, range(max(2, n_procs))))
    pids = sorted({pid for pid, _ in out})
    print(f"[MCMC-TEST] ok: unique_workers={len(pids)} pids={pids}", flush=True)

# ------------------------ Utility class ------------------------
class ClusterLensingUtils:
    def __init__(self, data: LensingData, z_s_ref: float, cosmo_H0: float = 70.0, cosmo_Om0: float = 0.3):
        self.data = data
        self.z_s_ref = z_s_ref

        self.alpha_maps_x_orig = data.alpha_maps_x
        self.alpha_maps_y_orig = data.alpha_maps_y
        self.lens_potential_maps_orig = data.lens_potential_maps

        self.map_sizes = [len(m) for m in data.alpha_maps_x]

        self._cosmo_cache = {}
        self.base_cosmo = self._get_cosmology(cosmo_H0, cosmo_Om0)

        self._x_grids: List[np.ndarray] = []
        for i in range(len(data.z_l_list)):
            size = self.map_sizes[i]
            pix = np.float32(data.pixscale[i])
            x_grid = np.linspace(0, size - 1, size, dtype=np.float32) * pix
            self._x_grids.append(x_grid)

        self._tls = threading.local()

        # Parent process caches
        self.ref_lens_models = []
        self.ref_solvers = []
        for i in range(len(data.z_l_list)):
            lens_model = LensModel(
                lens_model_list=['INTERPOL'],
                z_source=z_s_ref,
                z_lens=data.z_l_list[i],
                cosmo=self.base_cosmo
            )
            self.ref_lens_models.append(lens_model)
            self.ref_solvers.append(LensEquationSolver(lens_model))

        # Per-process caches
        self._creator_pid = os.getpid()
        self._proc_lens_models: Dict[int, LensModel] = {}
        self._proc_solvers: Dict[int, LensEquationSolver] = {}

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

    def _get_lens_and_solver(
        self,
        index: int,
        z_s_ref_override: Optional[float] = None,
        H0_override: Optional[float] = None,
    ) -> Tuple[LensModel, LensEquationSolver]:
        """
        Return a LensModel & LensEquationSolver built for the requested (z_s, H0).
        Caches are keyed by (index, z_s, H0) per-process to guarantee consistency.
        """
        # target redshift & H0
        z_src = float(self.z_s_ref if z_s_ref_override is None else z_s_ref_override)
        H0_use = float(self.base_cosmo.H0.value if H0_override is None else H0_override)

        pid = os.getpid()
        # per-process caches (dict init once)
        if not hasattr(self, "_proc_lens_models"):
            self._proc_lens_models = {}
            self._proc_solvers = {}

        # cache key (round to avoid float hash noise)
        key = (index, round(z_src, 8), round(H0_use, 8), pid)

        if key not in self._proc_lens_models:
            cosmo = self._get_cosmology(H0_use)
            lm = LensModel(
                lens_model_list=['INTERPOL'],
                z_source=z_src,
                z_lens=self.data.z_l_list[index],
                cosmo=cosmo,
            )
            self._proc_lens_models[key] = lm
            self._proc_solvers[key] = LensEquationSolver(lm)
            if os.environ.get("DEBUG_LENS"):
                print(f"[LensInit] pid={pid} built model/solver for cluster {index} z_s={z_src} H0={H0_use}", flush=True)

        return self._proc_lens_models[key], self._proc_solvers[key]


    def _get_scaled_model_and_kwargs(self, z_s: float, index: int, H0: float) -> Optional[Dict[str, Any]]:
        cosmo = self._get_cosmology(H0)
        D_S  = cosmo.angular_diameter_distance(z_s).value
        D_LS = cosmo.angular_diameter_distance_z1z2(self.data.z_l_list[index], z_s).value
        if D_S == 0 or not np.isfinite(D_S) or not np.isfinite(D_LS):
            return None

        scale = np.float32(D_LS / D_S)
        x_grid = self._x_grids[index]
        shape  = self.alpha_maps_x_orig[index].shape

        buf = self._get_thread_buffers(index, shape)
        np.multiply(self.lens_potential_maps_orig[index], scale, out=buf['f_'])
        np.multiply(self.alpha_maps_x_orig[index],        scale, out=buf['f_x'])
        np.multiply(self.alpha_maps_y_orig[index],        scale, out=buf['f_y'])

        return {
            'grid_interp_x': x_grid,
            'grid_interp_y': x_grid,
            'f_': buf['f_'],
            'f_x': buf['f_x'],
            'f_y': buf['f_y'],
        }


# ------------------------ Main class ------------------------
class ClusterLensing(ClusterLensingUtils):
    def calculate_images_and_delays(self, params: Dict[str, float], cluster_index: int) -> Dict[str, Any]:
        x_src, y_src = params["x_src"], params["y_src"]
        z_s  = params.get("z_s", self.z_s_ref)
        H0   = params.get("H0",  self.base_cosmo.H0.value)

        kwargs = self._get_scaled_model_and_kwargs(z_s, cluster_index, H0)
        if kwargs is None:
            print("Warning: Invalid cosmological parameters (e.g., z_s <= z_l).", flush=True)
            return {'image_positions': (np.array([]), np.array([])), 'time_delays': np.array([])}

        lens_model, solver = self._get_lens_and_solver(cluster_index, z_s_ref_override=z_s, H0_override=H0)

        x_img, y_img = solver.image_position_from_source(
            x_src, y_src, [kwargs],
            min_distance=self.data.pixscale[cluster_index],
            search_window=self.data.search_window_list[cluster_index],
            x_center=self.data.x_center[cluster_index],
            y_center=self.data.y_center[cluster_index],
        )

        if len(x_img) == 0:
            return {'image_positions': (np.array([]), np.array([])), 'time_delays': np.array([])}

        arrival_times = lens_model.arrival_time(x_img, y_img, [kwargs], x_source=x_src, y_source=y_src)
        return {'image_positions': (x_img, y_img), 'time_delays': np.sort(arrival_times - np.min(arrival_times))}


    def calculate_time_delay_uncertainty(self, img: np.ndarray, index: int) -> np.ndarray:
        sigma_dt = self.data.uncertainty_dt[index]
        pix = float(self.data.pixscale[index])
        x_pix = np.asarray(img[0], dtype=np.float64) / pix
        y_pix = np.asarray(img[1], dtype=np.float64) / pix

        n_x, n_y = sigma_dt.shape
        points = (np.arange(n_y), np.arange(n_x))
        xi = np.vstack((y_pix, x_pix)).T

        sigma_dt_values = interpn(points, sigma_dt, xi, method='linear', bounds_error=True)
        return np.asarray(sigma_dt_values)

    def _calculate_chi_squared(self, params: Dict[str, float], dt_true: np.ndarray, index: int,
                           sigma_lum: Optional[float] = None, lum_dist_true: Optional[float] = None) -> float:
        x_src, y_src = params["x_src"], params["y_src"]
        z_s  = params.get("z_s", self.z_s_ref)
        H0   = params.get("H0",  self.base_cosmo.H0.value)

        kwargs = self._get_scaled_model_and_kwargs(z_s, index, H0)
        if kwargs is None:
            return 5e5

        lens_model, solver = self._get_lens_and_solver(index, z_s_ref_override=z_s, H0_override=H0)

        x_img, y_img = solver.image_position_from_source(
            x_src, y_src, [kwargs],
            min_distance=self.data.pixscale[index],
            search_window=self.data.search_window_list[index],
            x_center=self.data.x_center[index],
            y_center=self.data.y_center[index],
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
        bounds = {
            "x_src": (self.data.x_center[index] - 50, self.data.x_center[index] + 50),
            "y_src": (self.data.y_center[index] - 50, self.data.y_center[index] + 50)
        }
        if fit_z:
            bounds["z_s"] = (1.0, 5.0)  # set to speed up convergence for testing
        if fit_hubble:
            bounds["H0"] = (60, 80)       # set to speed up convergence for testing
        _set_de_context(self, list(bounds.keys()), dt_true, index, sigma_lum, lum_dist_true)

        default_settings = {
            'strategy': 'rand1bin', 'maxiter': 200, 'popsize': 50, 'tol': 1e-7,
            'mutation': (0.5, 1), 'recombination': 0.7, 'polish': False,
            'updating': 'deferred',
            'disp': True, 'early_stop_threshold': 0.05
        }
        if de_settings:
            default_settings.update(de_settings)

        early_stop_threshold = default_settings.pop('early_stop_threshold', None)
        if early_stop_threshold is not None:
            def callback_fn(xk, convergence):
                return _de_objective_func_global(xk) < early_stop_threshold
            default_settings['callback'] = callback_fn

        # Linux fork default; cap workers to env or affinity
        try:
            slot_cpus = len(os.sched_getaffinity(0))
        except Exception:
            slot_cpus = os.cpu_count() or 1

        env_override = os.environ.get('DE_PROCS')
        if env_override:
            try:
                n_procs = max(1, int(env_override))
            except ValueError:
                n_procs = slot_cpus
            n_procs = min(n_procs, slot_cpus)
        else:
            n_procs = slot_cpus

        ctx = mp.get_context('fork') if platform.startswith('linux') else mp.get_context()
        print(f"[DE] slot_cpus={slot_cpus} env(DE_PROCS)={env_override} -> n_procs={n_procs}", flush=True)

        _parallel_self_test(n_procs, ctx)

        with ctx.Pool(
            processes=n_procs,
            initializer=_de_pool_initializer,
            initargs=(self, list(bounds.keys()), dt_true, index, sigma_lum, lum_dist_true),
        ) as pool:
            default_settings['updating'] = 'deferred'
            default_settings['disp'] = True
            result = differential_evolution(
                _de_objective_func_global,
                list(bounds.values()),
                workers=pool.map,
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
        n_steps   = mcmc_settings.get("n_steps", 10000)
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
        initial_state = np.array([initial_params[k] for k in bounds.keys()], dtype=float).reshape(-1)
        for j, k in enumerate(bounds.keys()):
            lo, hi = bounds[k]
            if not (lo <= initial_state[j] <= hi):
                raise ValueError(f"Initial parameter '{k}'={initial_state[j]:.3f} outside prior {bounds[k]}.")

        rng = np.random.default_rng()
        p0 = np.empty((n_walkers, ndim), dtype=float)
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

        env_override = os.environ.get('MCMC_PROCS')
        if env_override:
            try:
                n_procs = max(1, int(env_override))
            except ValueError:
                n_procs = slot_cpus
            n_procs = min(n_procs, slot_cpus)
        else:
            n_procs = slot_cpus

        ctx = mp.get_context('fork') if platform.startswith('linux') else mp.get_context()
        print(f"[MCMC] slot_cpus={slot_cpus} env(MCMC_PROCS)={env_override} -> n_procs={n_procs}", flush=True)

        _parallel_self_test_mcmc(n_procs, ctx)

        with ctx.Pool(
            processes=n_procs,
            initializer=_mcmc_pool_initializer,
            initargs=(self, dt_true, index, bounds, sigma_lum, lum_dist_true),
        ) as pool:
            sampler = emcee.EnsembleSampler(
                n_walkers, ndim, _log_posterior_global, pool=pool
            )
            sampler.run_mcmc(
                p0, n_steps,
                progress=True,
                progress_kwargs={"file": sys.stdout, "mininterval": 100}
            )

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
            print(f"--- Running DE for Cluster {i} ---", flush=True)
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
        print("\n--- DE Optimization Summary ---", flush=True)
        print(f"Best fit found for Cluster {best_result['cluster_index']} with chi^2 = {best_result['chi_sq']:.3f}", flush=True)
        print(f"Best-fit parameters: {best_result['de_params']}", flush=True)

        accepted_clusters = []
        if run_mcmc:
            mcmc_threshold = 2.3 + best_result['chi_sq']
            for result in results:
                if result['chi_sq'] <= mcmc_threshold:
                    accepted_clusters.append(result['cluster_index'])
                    print(f"\n--- Running MCMC for Cluster {result['cluster_index']} (chi_sq={result['chi_sq']:.3f}) ---", flush=True)
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
        print(f"Processing results for Cluster {best_result['cluster_index']}...", flush=True)

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
        print(f"-> Saved {flat_samples.shape[0]} samples to {output_path}", flush=True)

    def generate_source_positions(self,
                                  n_samples: int,
                                  index: int,
                                  search_range: float,
                                  img_no: int,
                                  H0: Optional[float] = None,
                                  z_s: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_center = self.data.x_center[index]
        y_center = self.data.y_center[index]

        n = 0
        x_srcs = np.array([])
        y_srcs = np.array([])
        z_ss = np.array([])
        H0s = np.array([])
        
        # Store initial arguments to avoid being overwritten in the loop
        H0_arg = H0
        z_s_arg = z_s
        
        while n < n_samples:
            x_src = np.random.uniform(x_center - search_range, x_center + search_range)
            y_src = np.random.uniform(y_center - search_range, y_center + search_range)
            
            current_z_s = z_s_arg if z_s_arg is not None else np.random.uniform(1.5, 4.0)
            current_H0 = H0_arg if H0_arg is not None else np.random.uniform(65, 76)

            test_params = {"x_src" : x_src, "y_src": y_src, "z_s": current_z_s, "H0": current_H0}
            test_cluster = index
            output = self.calculate_images_and_delays(test_params, test_cluster)
            if len(output['image_positions'][0]) == img_no:
                n += 1
                x_srcs = np.append(x_srcs, x_src)
                y_srcs = np.append(y_srcs, y_src)
                z_ss = np.append(z_ss, current_z_s)
                H0s = np.append(H0s, current_H0)
                print(x_src, y_src, current_z_s, current_H0, flush=True)

        return index, x_srcs, y_srcs, z_ss, H0s

#Tips now that it’s healthy:

# Turn off debug noise for a cleaner log: unset DEBUG_PAR, DEBUG_DE.

# If you want a tad more speed, switch DE updating back to 'deferred' (we set 'immediate' only for more frequent feedback).

# To double-confirm core usage on the slot:

# condor_ssh_to_job <cluster.proc> then run top/htop — you should see ~16 Python workers and ~1600% CPU total.

# ps -o pid,ppid,psr,comm -p <master_pid> --ppid <master_pid> to list worker PIDs and their CPUs.