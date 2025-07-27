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

def _de_objective_func(params: np.ndarray,
                       self_obj: 'ClusterLensing',
                       bounds_keys: List[str],
                       dt_true: np.ndarray,
                       index: int,
                       sigma_lum: Optional[float],
                       lum_dist_true: Optional[float]) -> float:
    """
    Global (picklable) objective function for differential_evolution.
    This function can be sent to worker processes.
    """
    param_map = {key: val for key, val in zip(bounds_keys, params)}
    return self_obj._calculate_chi_squared(
        param_map, dt_true, index, sigma_lum=sigma_lum, lum_dist_true=lum_dist_true
    )

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
    def calculate_images_and_delays(self, params: Dict[str, float], cluster_index: int) -> Dict[str, Any]:
        """
        Calculates image positions and time delays for a given parameter set.

        This is a public method for direct access to lensing calculations without
        running a full optimization.

        Args:
        ----
        params : Dict[str, float]
            A dictionary containing the parameters. Must include 'x_src' and 'y_src'.
            Can optionally include 'z_s' and 'H0' to override defaults.
        cluster_index : int
            The index of the cluster model to use for the calculation.

        Returns:
        -------
        Dict[str, Any]
            A dictionary containing the results, e.g.:
            {'image_positions': (np.ndarray, np.ndarray), 'time_delays': np.ndarray}
        """
        x_src, y_src = params["x_src"], params["y_src"]
        z_s = params.get("z_s", self.z_s_ref)
        H0 = params.get("H0", self.base_cosmo.H0.value)
        
        # Get the correctly scaled lens model for the given cosmology
        lens_model, kwargs = self._get_scaled_model_and_kwargs(z_s, cluster_index, H0)
        
        if lens_model is None:
            print("Warning: Invalid cosmological parameters (e.g., z_s <= z_l).")
            return {'image_positions': (np.array([]), np.array([])), 'time_delays': np.array([])}

        # Use lenstronomy's solver to find image positions
        solver = LensEquationSolver(lens_model)
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
            # Calculate arrival times for the found images
            arrival_times = lens_model.arrival_time(x_img, y_img, [kwargs], x_source=x_src, y_source=y_src)
            time_delays = arrival_times - np.min(arrival_times)
        
        return {'image_positions': (x_img, y_img), 'time_delays': np.sort(time_delays)}

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
            penalty = 3e3 # Penalties from original code
            return (abs(len(x_img) - len(dt_true)))**0.5 * penalty

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
        """
        # Define parameter bounds
        bounds = {
            "x_src": (self.data.x_center[index] - 50, self.data.x_center[index] + 50),
            "y_src": (self.data.y_center[index] - 50, self.data.y_center[index] + 50)
        }
        if fit_z:
            bounds["z_s"] = (1.0, 5.0)
        if fit_hubble:
            bounds["H0"] = (60, 80)
        
        # REMOVED: The local objective_func was here.

        # Package all additional arguments for the global objective function
        args_for_de = (
            self,
            list(bounds.keys()),
            dt_true,
            index,
            sigma_lum,
            lum_dist_true
        )

        # Default DE settings, can be overridden
        default_settings = {
            'strategy': 'rand1bin', 'maxiter': 200, 'popsize': 40, 'tol': 1e-7,
            'mutation': (0.5, 1), 'recombination': 0.7, 'polish': False,
            'updating': 'deferred', 'workers': -1, 'disp': True
        }
        if de_settings:
            default_settings.update(de_settings)

        early_stop_threshold = default_settings.pop('early_stop_threshold', 0.02) # Optional early stopping threshold
        if early_stop_threshold is not None:
            # The callback itself doesn't need to be pickled, so it can remain local.
            def callback_fn(xk, convergence):
                # It must call the new global objective function to check the value.
                current_chi_sq = _de_objective_func(xk, *args_for_de)
                if current_chi_sq < early_stop_threshold:
                    return True
                return False
            default_settings['callback'] = callback_fn

        # Call differential_evolution with the global function and args
        result = differential_evolution(
            _de_objective_func, 
            list(bounds.values()), 
            args=args_for_de,  # Pass the extra arguments here
            **default_settings
        )
        
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
        n_walkers = mcmc_settings.get("n_walkers", 32)
        n_steps = mcmc_settings.get("n_steps", 1000)
        sigma_dt = mcmc_settings.get("sigma_dt", 0.05)
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
        initial_state = np.array([initial_params[key] for key in bounds.keys()])

        # --- Defensive Check ---
        # First, check if the best-fit point itself is valid under the MCMC priors.
        for j, key in enumerate(bounds.keys()):
            if not (bounds[key][0] <= initial_state[j] <= bounds[key][1]):
                raise ValueError(
                    f"The initial best-fit parameter '{key}' with value {initial_state[j]:.3f} "
                    f"is outside the MCMC prior bounds of {bounds[key]}. "
                    "Check your mcmc_settings bounds."
                )

        # --- Corrected Walker Initialization ---
        # This robust loop is guaranteed to finish.
        p0 = np.zeros((n_walkers, ndim))
        print(f"Initializing {n_walkers} MCMC walkers...")
        for i in range(n_walkers):
            # Use a simple, clean `while True` loop
            while True:
                proposed_pos = initial_state + 1e-4 * np.random.randn(ndim)
                is_valid = True
                for j, key in enumerate(bounds.keys()):
                    if not (bounds[key][0] <= proposed_pos[j] <= bounds[key][1]):
                        is_valid = False
                        break  # This proposal is invalid, try again
                
                # If the proposal is valid, store it and break the inner loop
                if is_valid:
                    p0[i] = proposed_pos
                    break
        print("Walkers initialized successfully.")

        # Set up and run the sampler
        sampler = emcee.EnsembleSampler(
            n_walkers, ndim, _log_posterior_func,
            args=[self, dt_true, index, bounds, sigma_dt, sigma_lum, lum_dist_true]
        )
        # The progress bar will appear now
        sampler.run_mcmc(p0, n_steps, progress=True)
        return sampler

    def find_best_fit(self,
                      dt_true: np.ndarray,
                      run_mcmc: bool = False,
                      de_settings: Optional[Dict] = None,
                      mcmc_settings: Optional[Dict] = None) -> Tuple[List[Dict], List[int]]:
        """
        Main analysis pipeline.
        1. Iterates through all clusters and runs DE to find the best-fit cluster and parameters.
        2. Optionally runs MCMC for any cluster that is close to the best-fit chi-squared.
        """
        results = []
        mcmc_settings = mcmc_settings or {}
        # Determine what to fit based on settings
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
                lum_dist_true=lum_dist_true, # Added missing argument
                sigma_lum=sigma_lum,         # Added missing argument
                de_settings=de_settings
            )
            results.append({'cluster_index': i, 'params': best_params, 'chi_sq': min_chi_sq})
        
        # Find the best cluster based on chi-squared
        best_result = min(results, key=lambda x: x['chi_sq'])
        print("\n--- DE Optimization Summary ---")
        print(f"Best fit found for Cluster {best_result['cluster_index']} with chi^2 = {best_result['chi_sq']:.3f}")
        print(f"Best-fit parameters: {best_result['params']}")

        accepted_clusters = []
        if run_mcmc:
            # Define the chi-squared threshold based on the best DE result
            mcmc_threshold = 2.3 + best_result['chi_sq']
            
            # Run MCMC on all clusters that meet the criterion
            for result in results:
                if result['chi_sq'] <= mcmc_threshold:
                    accepted_clusters.append(result['cluster_index'])
                    print(f"\n--- Running MCMC for Cluster {result['cluster_index']} (chi_sq={result['chi_sq']:.3f}) ---")
                    sampler = self.run_mcmc_sampler(
                        dt_true,
                        result['cluster_index'],
                        result['params'],
                        mcmc_settings
                    )
                    result['mcmc_sampler'] = sampler

            # Keep only clusters that had MCMC sampling
            results = [res for res in results if 'mcmc_sampler' in res]
        
        return results, accepted_clusters
    
    def save_mcmc_results(self,
                          sampler: emcee.EnsembleSampler,
                          best_result: Dict,
                          n_burn_in: int,
                          output_path: str,
                          dt_true: np.ndarray,
                          mcmc_settings: Dict):
        """
        Processes the MCMC sampler results and saves them to a compressed NPZ file.

        Args:
            sampler (emcee.EnsembleSampler): The completed MCMC sampler object.
            best_result (Dict): The dictionary containing the best-fit cluster info.
            n_burn_in (int): The number of burn-in steps to discard.
            output_path (str): The path to the output file (e.g., 'results/cluster_0_posterior.npz').
            dt_true (np.ndarray): The true time delays, for chi-squared calculation.
            mcmc_settings (Dict): The MCMC settings dictionary, for cosmology info.
        """
        print(f"Processing results for Cluster {best_result['cluster_index']}...")

        # Get the chains and log probabilities, discarding burn-in
        flat_samples = sampler.get_chain(discard=n_burn_in, flat=True)
        log_probs = sampler.get_log_prob(discard=n_burn_in, flat=True)
        full_chain = sampler.get_chain()

        # Calculate the median values for each parameter
        medians_array = np.median(flat_samples, axis=0)
        
        # Reconstruct the parameter dictionary from the median values
        param_labels = list(best_result['params'].keys())
        median_params = {key: val for key, val in zip(param_labels, medians_array)}

        # Calculate the chi-squared value for the median-fit parameters
        # This uses the same internal function as the MCMC for consistency.
        chi_sq_final = self._calculate_chi_squared(
            params=median_params,
            dt_true=dt_true,
            index=best_result['cluster_index'],
            sigma_dt=mcmc_settings.get("sigma_dt", 0.05),
            sigma_lum=mcmc_settings.get("sigma_lum"),
            lum_dist_true=mcmc_settings.get("lum_dist_true")
        )
        
        # Save the data to a compressed NumPy file
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
    
    # Utility functions requiring main analysis functions
    def generate_source_positions(self, n_samples: int, index: int, search_range: float, img_no: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates random source positions within the cluster's search window.
        
        Args:
        ----
        n_samples : int
            Number of random source positions to generate.
        index : int
            Index of the cluster model to search suitable source positions.
        search_range : float
            The search window around the cluster center to generate source positions.
        img_no : int
            Image number (for future use, if needed).

        Returns:
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Arrays of cluster index and x, y source positions.
        """
        x_center = self.data.x_center[index]
        y_center = self.data.y_center[index]
        
        n = 0
        x_srcs = np.array([])
        y_srcs = np.array([])
        z_ss = np.array([])
        H0s = np.array([])
        while n < n_samples:
            x_src = np.random.uniform(x_center - search_range, x_center + search_range)
            y_src = np.random.uniform(y_center - search_range, y_center + search_range)
            z_s = np.random.uniform(2.0, 4.0)  # Random redshift for the source
            H0 = np.random.uniform(65, 76)  # Random Hubble constant

            test_params = {"x_src" : x_src, "y_src": y_src, "z_s": z_s, "H0": H0}
            test_cluster = index
            output = self.calculate_images_and_delays(
                test_params, test_cluster
            )
            # Check if the number of images matches the expected count
            if len(output['image_positions'][0]) >= img_no:
                n += 1
                x_srcs = np.append(x_srcs, x_src)
                y_srcs = np.append(y_srcs, y_src)
                z_ss = np.append(z_ss, z_s)
                H0s = np.append(H0s, H0)

        return index, x_srcs, y_srcs, z_ss, H0s
