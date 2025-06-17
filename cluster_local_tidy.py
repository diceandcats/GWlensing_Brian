# pylint: skip-file
# cluster_lensing_system.py
import numpy as np
from scipy.optimize import differential_evolution
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import emcee
from typing import List, Optional, Tuple, Dict, Any

# Assuming lensing_data.py is in the same directory or accessible
from lensing_data_class import LensingData

# --- Global Function for MCMC Parallelization ---

def _log_posterior_func(params: np.ndarray,
                        self_obj: 'ClusterLensing',
                        dt_true: np.ndarray,
                        index: int,
                        bounds: Dict[str, Tuple[float, float]],
                        sigma_dt: float,
                        sigma_lum: Optional[float],
                        lum_dist_true: Optional[float]) -> float:
    """
    Global (picklable) log-posterior function for emcee.
    Handles priors and calls the main chi-squared calculation.
    """
    param_map = {"x_src": params[0], "y_src": params[1]}
    fit_z = "z_s" in bounds
    fit_hubble = "H0" in bounds

    if fit_z:
        param_map["z_s"] = params[2]
        if fit_hubble:
            param_map["H0"] = params[3]
    elif fit_hubble:
        param_map["H0"] = params[2]

    # 1) Uniform prior checks
    for i, key in enumerate(bounds.keys()):
        if not (bounds[key][0] <= params[i] <= bounds[key][1]):
            return -np.inf

    # 2) Log-likelihood = -0.5 * chi^2
    chi_sq = self_obj._calculate_chi_squared(
        param_map, dt_true, index, sigma_dt, sigma_lum, lum_dist_true
    )
    
    return -0.5 * chi_sq

# --- Utility Class ---

class ClusterLensingUtils:
    """
    Handles data initialization, cosmological calculations, and lensing map scaling.
    """
    def __init__(self, data: LensingData, z_s_ref: float, cosmo_H0: float = 70.0, cosmo_Om0: float = 0.3):
        self.data = data
        self.z_s_ref = z_s_ref
        
        # Store original maps safely
        self.alpha_maps_x_orig = [np.copy(m) for m in data.alpha_maps_x]
        self.alpha_maps_y_orig = [np.copy(m) for m in data.alpha_maps_y]
        self.lens_potential_maps_orig = [np.copy(m) for m in data.lens_potential_maps]
        
        self.map_sizes = [len(m) for m in data.alpha_maps_x]
        
        # Caching for cosmology objects to prevent re-creation
        self._cosmo_cache = {}
        self.base_cosmo = self._get_cosmology(cosmo_H0, cosmo_Om0)

        # Pre-calculate lenstronomy components for the reference redshift
        self.ref_lens_models = []
        self.ref_solvers = []
        self.ref_kwargs_list = []
        for i in range(len(data.z_l_list)):
            x_grid = np.linspace(0, self.map_sizes[i] - 1, self.map_sizes[i]) * data.pixscale[i]
            kwargs = {
                'grid_interp_x': x_grid, 'grid_interp_y': x_grid,
                'f_': data.lens_potential_maps[i] * data.pixscale[i]**2,
                'f_x': data.alpha_maps_x[i], 'f_y': data.alpha_maps_y[i]
            }
            self.ref_kwargs_list.append(kwargs)
            
            lens_model = LensModel(lens_model_list=['INTERPOL'], z_source=z_s_ref, z_lens=data.z_l_list[i], cosmo=self.base_cosmo)
            self.ref_lens_models.append(lens_model)
            self.ref_solvers.append(LensEquationSolver(lens_model))

    def _get_cosmology(self, H0: float, Om0: float = 0.3) -> FlatLambdaCDM:
        """Returns a cached or new cosmology instance."""
        if H0 not in self._cosmo_cache:
            self._cosmo_cache[H0] = FlatLambdaCDM(H0=H0, Om0=Om0)
        return self._cosmo_cache[H0]

    def _get_scaled_model_and_kwargs(self, z_s: float, index: int, H0: float) -> Tuple[Optional[LensModel], Optional[Dict[str, Any]]]:
        """
        Calculates the scaled lens model and kwargs for a given z_s and H0.
        Returns (None, None) if scaling is not possible.
        """
        cosmo = self._get_cosmology(H0)
        
        D_S = cosmo.angular_diameter_distance(z_s).value
        D_LS = cosmo.angular_diameter_distance_z1z2(self.data.z_l_list[index], z_s).value
        
        if D_S == 0:
            return None, None
            
        scale = D_LS / D_S
        pix = self.data.pixscale[index]
        size = self.map_sizes[index]
        x_grid = np.linspace(0, size - 1, size) * pix
        
        scaled_kwargs = {
            'grid_interp_x': x_grid,
            'grid_interp_y': x_grid,
            'f_': self.lens_potential_maps_orig[index] * scale * pix**2,
            'f_x': self.alpha_maps_x_orig[index] * scale,
            'f_y': self.alpha_maps_y_orig[index] * scale
        }
        
        lens_model = LensModel(lens_model_list=['INTERPOL'], z_source=z_s, z_lens=self.data.z_l_list[index], cosmo=cosmo)
        
        return lens_model, scaled_kwargs

# --- Main Analysis Class ---

class ClusterLensing(ClusterLensingUtils):
    """
    Main class for running lensing analysis, including optimization and MCMC.
    Inherits data handling and basic calculations from ClusterLensingUtils.
    """
    def _calculate_chi_squared(self,
                             params: Dict[str, float],
                             dt_true: np.ndarray,
                             index: int,
                             sigma_dt: float = 0.05,
                             sigma_lum: Optional[float] = None,
                             lum_dist_true: Optional[float] = None) -> float:
        """
        Unified chi-squared function for DE optimization and MCMC likelihood.
        """
        x_src, y_src = params["x_src"], params["y_src"]
        z_s = params.get("z_s", self.z_s_ref)
        H0 = params.get("H0", self.base_cosmo.H0.value)
        
        # Get the appropriate lens model and parameters
        lens_model, kwargs = self._get_scaled_model_and_kwargs(z_s, index, H0)
        if lens_model is None:
            return 5e5  # Penalty for invalid cosmology

        # 1. Find image positions
        solver = LensEquationSolver(lens_model)
        x_img, y_img = solver.image_position_from_source(
            x_src, y_src, [kwargs],
            min_distance=self.data.pixscale[index],
            search_window=self.data.search_window_list[index],
            x_center=self.data.x_center[index],
            y_center=self.data.y_center[index]
        )
        
        # 2. Check image count and apply penalties
        if len(x_img) != len(dt_true):
            penalty = {0: 6.7e3, 1: 3e3} # Penalties from original code
            return (abs(len(x_img) - len(dt_true)))**0.5 * penalty.get(len(x_img), 3e3)

        # 3. Calculate time delays and time-delay chi-squared
        t = lens_model.arrival_time(x_img, y_img, [kwargs], x_source=x_src, y_source=y_src)
        dt_candidate = t - t.min()
        
        mask = np.array(dt_true) != 0
        sigma_arr = sigma_dt * np.array(dt_true)
        chi_sq_dt = np.sum((dt_candidate[mask] - dt_true[mask])**2 / sigma_arr[mask]**2)
        
        # 4. Add luminosity distance chi-squared if applicable
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

        Returns:
        --------
        A tuple containing:
        - A dictionary with the best-fit parameters (e.g., {'x_src': ..., 'y_src': ...}).
        - The minimum chi-squared value.
        """
        # Define parameter bounds
        bounds = {
            "x_src": (self.data.x_center[index] - 50, self.data.x_center[index] + 50),
            "y_src": (self.data.y_center[index] - 50, self.data.y_center[index] + 50)
        }
        if fit_z:
            bounds["z_s"] = (1.0, 5.0)
        if fit_hubble:
            bounds["H0"] = (53, 92)

        # Objective function for scipy
        def objective_func(params):
            param_map = {key: val for key, val in zip(bounds.keys(), params)}
            return self._calculate_chi_squared(param_map, dt_true, index, sigma_lum=sigma_lum, lum_dist_true=lum_dist_true)
        
        # Default DE settings, can be overridden
        default_settings = {
            'strategy': 'rand1bin', 'maxiter': 200, 'popsize': 40, 'tol': 1e-7,
            'mutation': (0.5, 1), 'recombination': 0.7, 'polish': False,
            'updating': 'deferred', 'workers': -1, 'disp': True
        }
        if de_settings:
            default_settings.update(de_settings)

        early_stop_threshold = default_settings.pop('early_stop_threshold', None)
        if early_stop_threshold is not None:
            def callback_fn(xk, convergence):
                if objective_func(xk) < early_stop_threshold:
                    return True
                return False
            default_settings['callback'] = callback_fn

        result = differential_evolution(objective_func, list(bounds.values()), **default_settings)
        
        best_params = {key: val for key, val in zip(bounds.keys(), result.x)}
        return best_params, result.fun

    def run_mcmc_sampler(self,
                         dt_true: np.ndarray,
                         index: int,
                         initial_params: Dict[str, float],
                         mcmc_settings: Dict[str, Any]) -> emcee.EnsembleSampler:
        """
        Runs emcee MCMC sampler to explore the posterior distribution.
        """
        # Unpack settings with defaults
        n_walkers = mcmc_settings.get("n_walkers", 32)
        n_steps = mcmc_settings.get("n_steps", 1000)
        sigma_dt = mcmc_settings.get("sigma_dt", 0.05)
        sigma_lum = mcmc_settings.get("sigma_lum", 0.05)
        lum_dist_true = mcmc_settings.get("lum_dist_true")

        # Define parameter bounds for prior
        bounds = {
            "x_src": mcmc_settings.get("x_bounds", (initial_params["x_src"] - 5, initial_params["x_src"] + 5)),
            "y_src": mcmc_settings.get("y_bounds", (initial_params["y_src"] - 5, initial_params["y_src"] + 5)),
        }
        if "z_s" in initial_params:
            bounds["z_s"] = mcmc_settings.get("z_bounds", (1.0, 5.0))
        if "H0" in initial_params:
             bounds["H0"] = mcmc_settings.get("H0_bounds", (53, 92))
        
        ndim = len(bounds)
        
        # Initialize walkers in a small ball around the initial guess
        initial_state = np.array([initial_params[key] for key in bounds.keys()])
        p0 = initial_state + 1e-4 * np.random.randn(n_walkers, ndim)

        # Set up and run the sampler
        sampler = emcee.EnsembleSampler(
            n_walkers, ndim, _log_posterior_func,
            args=[self, dt_true, index, bounds, sigma_dt, sigma_lum, lum_dist_true]
        )
        sampler.run_mcmc(p0, n_steps, progress=True)
        return sampler

    def find_best_fit(self,
                      dt_true: np.ndarray,
                      run_mcmc: bool = False,
                      de_settings: Optional[Dict] = None,
                      mcmc_settings: Optional[Dict] = None) -> List[Dict]:
        """
        Main analysis pipeline.
        1. Iterates through all clusters and runs DE to find the best-fit cluster and parameters.
        2. Optionally runs MCMC for the best-fit cluster to sample the posterior.
        """
        results = []
        # Determine what to fit based on settings
        fit_z = mcmc_settings.get("fit_z", False) if mcmc_settings else False
        fit_hubble = mcmc_settings.get("fit_hubble", False) if mcmc_settings else False

        for i in range(len(self.data.z_l_list)):
            print(f"--- Running DE for Cluster {i} ---")
            best_params, min_chi_sq = self.run_de_optimization(
                dt_true, i, fit_z=fit_z, fit_hubble=fit_hubble, de_settings=de_settings
            )
            results.append({'cluster_index': i, 'params': best_params, 'chi_sq': min_chi_sq})
        
        # Find the best cluster based on chi-squared
        best_result = min(results, key=lambda x: x['chi_sq'])
        print("\n--- DE Optimization Summary ---")
        print(f"Best fit found for Cluster {best_result['cluster_index']} with chi^2 = {best_result['chi_sq']:.3f}")
        print(f"Best-fit parameters: {best_result['params']}")

        if run_mcmc:
            print(f"\n--- Running MCMC for Best-Fit Cluster {best_result['cluster_index']} ---")
            sampler = self.run_mcmc_sampler(
                dt_true,
                best_result['cluster_index'],
                best_result['params'],
                mcmc_settings or {}
            )
            best_result['mcmc_sampler'] = sampler

        return best_result