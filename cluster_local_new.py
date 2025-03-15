import numpy as np
from scipy.optimize import minimize, differential_evolution
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import emcee
from emcee.moves import StretchMove
import multiprocessing as mp

# pylint: disable=C0103

def _log_posterior_func(params,
                        x_center, y_center, x_range, y_range,
                        self_obj,
                        dt_true, index, sigma,
                        fix_z, z_s_fix,
                        z_lower=0, z_upper=100):
    """
    Global (picklable) log-posterior function for emcee.

    Parameters
    ----------
    params : array-like
        The parameter vector. Normally (x_src, y_src, z_s), but if fix_z=True then (x_src, y_src).
    x_center, y_center : float
        Center coordinates of the lens for bounding the x,y prior.
    x_range, y_range : float
        x in [x_center - x_range, x_center + x_range], similarly for y.
    self_obj : ClusterLensing_fyp
        Instance of your class. We'll call self_obj.chi_squared_with_z(...) on it.
    dt_true : array-like
        Observed time delays.
    index : int
        Lens index.
    sigma : float
        Uncertainty for chi^2.
    fix_z : bool
        If True, z is held fixed; otherwise it's a free parameter.
    z_s_fix : float
        The fixed source redshift if fix_z=True.
    z_lower, z_upper : float
        If fix_z=False, then z_s in [z_lower, z_upper].

    Returns
    -------
    float
        log-posterior value (log-likelihood + log-prior).
    """

    # 1) Unpack params differently if fix_z is True
    if fix_z:
        x_src, y_src = params
        z_s = z_s_fix
    else:
        x_src, y_src, z_s = params

    # 2) Uniform prior checks
    in_x = (x_center - x_range <= x_src <= x_center + x_range)
    in_y = (y_center - y_range <= y_src <= y_center + y_range)
    in_z = (z_lower <= z_s <= z_upper)
    if not (in_x and in_y and in_z):
        return -np.inf

    # 3) Log-likelihood = -0.5 * chi^2
    chi_sq = self_obj.chi_squared_with_z((x_src, y_src, z_s),
                                         dt_true, index=index, sigma=sigma)
    log_likelihood = -0.5 * chi_sq

    # Uniform prior => log_prior = 0 in these bounds
    log_prior = 0.0
    return log_prior + log_likelihood

class ClusterLensing_fyp:
    """
    Class for localization.
    """
    def __init__(self, alpha_maps_x, alpha_maps_y, lens_potential_maps, z_l_default,
                 z_s, pixscale, diff_z=False):
        """
        Parameters:
        ---------------
        alpha_maps_x: List of 6 deflection maps in x direction in arcsec.
        alpha_maps_y: List of 6 deflection maps in y direction in arcsec.
        lens_potential_maps: List of 6 lens potential maps in arcsec^2.
        z_l: The redshift of the lens.
        z_s: The redshift of the source.
        pixscale: The list of pixel scales for each cluster.
        diff_z: Boolean indicating if differential redshift scaling is applied.
        """
        self.alpha_maps_x = alpha_maps_x    # List of arrays
        self.alpha_maps_y = alpha_maps_y    # List of arrays
        self.lens_potential_maps = lens_potential_maps  # List of arrays
        self.z_l = z_l_default
        self.z_s = z_s
        self.pixscale = pixscale
        self.image_positions = None
        self.magnifications = None
        self.time_delays = None
        self.diff_z = diff_z
        self.scale_factor = None  # for scaling the deflection maps

        # --- Modification 1: Cache the cosmology instance ---
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        # Save copies of the original maps for re-scaling
        self.alpha_maps_x_orig = [np.copy(m) for m in alpha_maps_x]
        self.alpha_maps_y_orig = [np.copy(m) for m in alpha_maps_y]
        self.lens_potential_maps_orig = [np.copy(m) for m in lens_potential_maps]

        self.x_center, self.y_center = [90, 75, 110, 70, 90, 70], [70, 80, 95, 60, 93, 65]
        self.search_window_list = [90.1, 95, 100, 85, 100, 90]
        self.z_l_list = [0.375, 0.308, 0.351, 0.397, 0.545, 0.543]

        # Get the size of the deflection maps
        self.size = []
        for i in range(6):
            self.size.append(len(alpha_maps_x[i]))

        # Default lens model without scaling
        self.lens_models = []
        self.solvers = []
        self.kwargs_list = []
        for i in range(6):
            x_grid = np.linspace(0, self.size[i] - 1, self.size[i]) * pixscale[i]
            kwargs = {
                'grid_interp_x': x_grid,
                'grid_interp_y': x_grid,
                'f_': lens_potential_maps[i] * pixscale[i]**2,
                'f_x': alpha_maps_x[i],
                'f_y': alpha_maps_y[i]
            }
            self.kwargs_list.append(kwargs)
            lens_model_list = ['INTERPOL']
            lensmodel = LensModel(lens_model_list=lens_model_list, z_source=z_s, z_lens=z_l_default)
            solver = LensEquationSolver(lensmodel)
            self.lens_models.append(lensmodel)
            self.solvers.append(solver)

        if diff_z:
            self.D_S1, self.D_S2, self.D_LS1, self.D_LS2 = self.scaling()

    def scaling(self):
        """
        Scale the deflection maps (and potentially the lens potential maps) from a reference source redshift.
        """
        z_L = self.z_l
        z_S = self.z_s

        D_S1 = self.cosmo.angular_diameter_distance(1.0)
        D_S2 = self.cosmo.angular_diameter_distance(z_S)
        D_LS1 = self.cosmo.angular_diameter_distance_z1z2(z_L, 1.0)
        D_LS2 = self.cosmo.angular_diameter_distance_z1z2(z_L, z_S)

        scal = (D_LS1 * D_S2) / (D_LS2 * D_S1)

        for i in range(6):
            self.alpha_maps_x[i] *= scal
            self.alpha_maps_y[i] *= scal

        return D_S1, D_S2, D_LS1, D_LS2

    def image_position(self, x_src, y_src, index=0):
        solver = self.solvers[index]
        kwargs_lens = [self.kwargs_list[index]]
        image_positions = solver.image_position_from_source(
            x_src, y_src, kwargs_lens,
            min_distance=self.pixscale[index],
            search_window=self.search_window_list[int(index)],
            verbose=False,
            x_center=self.x_center[int(index)],
            y_center=self.y_center[int(index)]
        )
        return image_positions

    # --- Modification 2: Optimized image_position_z with candidate_kwargs ---
    def image_position_z(self, x_src, y_src, z, index=0, candidate_kwargs=None):
        """
        Returns the image positions at source (x_src, y_src) for a given candidate source redshift z.
        If candidate_kwargs is provided, it is used instead of the stored self.kwargs_list.
        """
        lens_model_z = LensModel(lens_model_list=['INTERPOL'], z_source=z, z_lens=self.z_l_list[index])
        solver_z = LensEquationSolver(lens_model_z)
        kwargs_lens = [candidate_kwargs] if candidate_kwargs is not None else [self.kwargs_list[index]]
        image_positions = solver_z.image_position_from_source(
            x_src, y_src, kwargs_lens,
            min_distance=self.pixscale[index],
            search_window=self.search_window_list[int(index)],
            verbose=False,
            x_center=self.x_center[int(index)],
            y_center=self.y_center[int(index)]
        )
        return image_positions

    # def rand_src_test(self, n_each=10, n=5):
    #     srcs = []
    #     indices = []
    #     clusters = 6  # Number of clusters

    #     for cluster_index in range(clusters):
    #         count = 0
    #         while count < n_each:
    #             x_center = self.x_center[int(cluster_index)]
    #             x_src = np.random.uniform(x_center - 30, x_center + 30)
    #             y_center = self.y_center[int(cluster_index)]
    #             y_src = np.random.uniform(y_center - 30, y_center + 30)
    #             img_positions = self.image_position(x_src, y_src, cluster_index)
    #             if len(img_positions[0]) >= n:
    #                 print(f"Cluster index: {cluster_index}, Number of images: {len(img_positions[0])}")
    #                 srcs.append((x_src, y_src))
    #                 indices.append(cluster_index)
    #                 count += 1
    #     return srcs, indices

    # def rand_src_test_1_cluster(self, n_each=50, n=5, index=0):
    #     srcs = []
    #     indices = []
    #     count = 0
    #     while count < n_each:
    #         x_center = self.x_center[int(index)]
    #         x_src = np.random.uniform(x_center - 30, x_center + 30)
    #         y_center = self.y_center[int(index)]
    #         y_src = np.random.uniform(y_center - 30, y_center + 30)
    #         img_positions = self.image_position(x_src, y_src, index)
    #         if len(img_positions[0]) >= n:
    #             print(f"Cluster index: {index}, Number of images: {len(img_positions[0])}")
    #             srcs.append((x_src, y_src))
    #             indices.append(index)
    #             count += 1
    #     return srcs, indices

    # def correct_indices_first(self, df):
    #     corrected_indices = []
    #     for idx, row in tqdm(df.iterrows()):
    #         x = row['x']
    #         y = row['y']
    #         indices = int(row['indices'])
    #         image_positions = self.image_position(x, y, indices)
    #         no_images = len(image_positions[0])
    #         success = False
    #         if no_images < 5:
    #             for possible_index in range(6):
    #                 image_positions = self.image_position(x, y, possible_index)
    #                 if len(image_positions[0]) >= 5:
    #                     corrected_indices.append(possible_index)
    #                     success = True
    #                     break
    #             if not success:
    #                 print(f'error in row {idx}')
    #                 corrected_indices.append(indices)
    #         else:
    #             corrected_indices.append(indices)
    #     df['indices'] = corrected_indices
    #     return df

    def time_delay(self, x_img, y_img, index=0, x_src=None, y_src=None):
        kwargs = self.kwargs_list[index]
        lens_model = self.lens_models[index]
        t = lens_model.arrival_time(x_img, y_img, [kwargs], x_source=x_src, y_source=y_src)
        dt = t - t.min()
        return dt
    
    def time_delay_z(self, x_img, y_img, z_s, index=0, x_src=None, y_src=None):
        D_S_candidate = self.cosmo.angular_diameter_distance(z_s)
        D_LS_candidate = self.cosmo.angular_diameter_distance_z1z2(self.z_l_list[index], z_s)
        candidate_scale = D_LS_candidate / D_S_candidate

        size = self.size[index]
        pix = self.pixscale[index]
        x_grid = np.linspace(0, size - 1, size) * pix
        alpha_x = self.alpha_maps_x_orig[index] * candidate_scale
        alpha_y = self.alpha_maps_y_orig[index] * candidate_scale
        potential = self.lens_potential_maps_orig[index] * candidate_scale
        candidate_kwargs_2 = {
            'grid_interp_x': x_grid,
            'grid_interp_y': x_grid,
            'f_': potential * pix**2,
            'f_x': alpha_x,
            'f_y': alpha_y
        }

        lens_model_2 = LensModel(lens_model_list=['INTERPOL'], z_source=z_s, z_lens=self.z_l_list[index])
        t = lens_model_2.arrival_time(x_img, y_img, [candidate_kwargs_2],
                                               x_source=x_src, y_source=y_src)
        dt = t - t.min()
        return dt
    
    def my_image_and_delay_for_xyz(self, x_src, y_src, z_s, index=0):
        # 1) Re-scale the deflection/potential maps for the new z_s
        D_S_candidate = self.cosmo.angular_diameter_distance(z_s)
        D_LS_candidate = self.cosmo.angular_diameter_distance_z1z2(self.z_l_list[index], z_s)
        candidate_scale = D_LS_candidate / D_S_candidate
        
        size = self.size[index]
        pix = self.pixscale[index]
        x_grid = np.linspace(0, size - 1, size) * pix
        
        candidate_alpha_x = self.alpha_maps_x_orig[index] * candidate_scale
        candidate_alpha_y = self.alpha_maps_y_orig[index] * candidate_scale
        candidate_potential = self.lens_potential_maps_orig[index] * candidate_scale
        
        candidate_kwargs = {
            'grid_interp_x': x_grid,
            'grid_interp_y': x_grid,
            'f_': candidate_potential * pix**2,
            'f_x': candidate_alpha_x,
            'f_y': candidate_alpha_y
        }

        # 2) Solve for image positions, using that candidate_kwargs
        x_img, y_img = self.image_position_z(x_src, y_src, z_s,
                                            index=index,
                                            candidate_kwargs=candidate_kwargs)
        
        # 3) Then compute time delay
        dt = self.time_delay_z(x_img, y_img, z_s,
                            index=index,
                            x_src=x_src,
                            y_src=y_src)
        return (x_img, y_img, dt)

    def chi_squared(self, src_guess, dt_true, index=0, sigma=0.05):
        x_src, y_src = src_guess
        img = self.image_position(x_src, y_src, index)
        if len(img[0]) == 0:
            return 1e13
        if len(img[0]) != len(dt_true):
            return abs(len(img[0]) - len(dt_true)) * 1.4e12
        t = self.time_delay(img[0], img[1], index, x_src, y_src)
        dt = t - t.min()
        chi_sq = np.sum((dt - dt_true) ** 2) / (2 * sigma ** 2)
        return chi_sq

    # --- Modification 3: Optimized chi_squared_with_z using cached cosmology ---
    def chi_squared_with_z(self, src_guess, dt_true, index=0, sigma=0.05):
        """
        Simplified chi-squared function: candidate scale is computed as
            candidate_scale = D_LS(z_l, z) / D_S(z)
        because the original deflection maps are normalized to D_LS(z_l, z_s)/D_S(z_s)=1.
        Scaling is performed here, and image positions are calculated using image_position_z.
        """
        x_src, y_src,z_s = src_guess

        D_S_candidate = self.cosmo.angular_diameter_distance(z_s)
        D_LS_candidate = self.cosmo.angular_diameter_distance_z1z2(self.z_l_list[index], z_s)
        if D_S_candidate == 0 or D_LS_candidate == 0:
            return 1e6
        candidate_scale = D_LS_candidate / D_S_candidate

        size = self.size[index]
        pix = self.pixscale[index]
        x_grid = np.linspace(0, size - 1, size) * pix
        candidate_alpha_x = self.alpha_maps_x_orig[index] * candidate_scale
        candidate_alpha_y = self.alpha_maps_y_orig[index] * candidate_scale
        candidate_potential = self.lens_potential_maps_orig[index] * candidate_scale
        candidate_kwargs = {
            'grid_interp_x': x_grid,
            'grid_interp_y': x_grid,
            'f_': candidate_potential * pix**2,
            'f_x': candidate_alpha_x,
            'f_y': candidate_alpha_y
        }

        img = self.image_position_z(x_src, y_src, z_s, index=index, candidate_kwargs=candidate_kwargs)
        if len(img[0]) == 0:
            return 3.5e5
        if len(img[0]) != len(dt_true):
            return abs(len(img[0]) - len(dt_true)) * 5e4

        candidate_lens_model = LensModel(lens_model_list=['INTERPOL'], z_source=z_s, z_lens=self.z_l_list[index])
        t = candidate_lens_model.arrival_time(img[0], img[1], [candidate_kwargs],
                                               x_source=x_src, y_source=y_src)
        dt_candidate = t - t.min()
        sigma_arr = sigma * np.array(dt_true)
        mask = np.array(dt_true) != 0
        chi_sq = np.sum((dt_candidate[mask] - dt_true[mask])**2 / sigma_arr[mask]**2) / 2
        return chi_sq

    def localize_known_cluster_diffevo(self, dt_true, index=1):
        x_center = self.x_center[int(index)]
        y_center = self.y_center[int(index)]
        x_min, x_max = x_center - 50, x_center + 50
        y_min, y_max = y_center - 50, y_center + 50
        bounds = [(x_min, x_max), (y_min, y_max)]

        def callback_fn(xk, convergence):
            if self.chi_squared(xk, dt_true, index) < 1e-2:
                return True
            return False

        result = differential_evolution(
            self.chi_squared,
            bounds,
            args=(dt_true, index),
            strategy='rand1bin',
            maxiter=250,
            popsize=40,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            polish=False,
            updating='deferred',
            workers=-1,
            disp=True,
            callback=callback_fn
        )
        x_opt, y_opt = result.x
        min_chi_sq = result.fun
        return x_opt, y_opt, min_chi_sq

    def localize_known_cluster_diffevo_with_z(self, dt_true, index=1, threshold=1e-2):
        x_center = self.x_center[int(index)]
        y_center = self.y_center[int(index)]
        x_min, x_max = x_center - 50, x_center + 50
        y_min, y_max = y_center - 50, y_center + 50
        z_lower = 2.0
        z_upper = 3.5
        bounds = [(x_min, x_max), (y_min, y_max), (z_lower, z_upper)]

        # def objective(params):
        #     x_src, y_src, z_candidate = params
        #     return self.chi_squared_with_z((x_src, y_src), z_candidate, dt_true, index=index, sigma=sigma)

        def callback_fn(xk, convergence):
            if self.chi_squared_with_z(xk, dt_true, index) < threshold:
                return True
            return False

        result = differential_evolution(
            self.chi_squared_with_z,
            bounds,
            args=(dt_true, index),
            strategy='rand1bin',
            maxiter=150,
            popsize=40,
            tol=1e-7,
            mutation=(0.5, 1),
            recombination=0.7,
            polish=False,
            updating='deferred',
            workers=-1,
            disp=True,
            callback=callback_fn
            )
        
        x_opt, y_opt, z_opt = result.x
        min_chi_sq = result.fun

        if min_chi_sq > 5 * threshold:
            return None, None, None, None

        return x_opt, y_opt, z_opt, min_chi_sq

    def localize_diffevo(self, dt_true):
        chi_sqs = []
        src_guess = []
        for i in range(6):
            x_guess, y_guess, chi_sq = self.localize_known_cluster_diffevo(dt_true, i)
            src_guess.append([x_guess, y_guess])
            chi_sqs.append(chi_sq)
        min_chi_sq = min(chi_sqs)
        index = chi_sqs.index(min_chi_sq)
        return index, src_guess[index], min_chi_sq, src_guess, chi_sqs
    
    def localize_diffevo_with_z(self, dt_true):
        chi_sqs = []
        src_guess = []
        for i in range(6):
            x_guess, y_guess, z_guess, chi_sq = self.localize_known_cluster_diffevo_with_z(dt_true, i)
            src_guess.append([x_guess, y_guess, z_guess])
            chi_sqs.append(chi_sq)
        min_chi_sq = min(chi_sqs)
        index = chi_sqs.index(min_chi_sq)
        return index, src_guess[index], min_chi_sq, src_guess, chi_sqs
    
    
    # def localize_mcmc_emcee_fix_z(self, dt_true, index=0,
    #                               n_walkers=50, n_steps=5000, burn_in=1000,
    #                               x_range=50.0, y_range=50.0,
    #                               z_s_fix=1.0,  # your fixed redshift
    #                               sigma=0.05,
    #                               random_seed=42,
    #                               n_processes=8, fix_z=True):
    #     """
    #     Run MCMC with x_src,y_src as free parameters, and z_s = z_s_fix (held fixed).
    #     """
    #     np.random.seed(random_seed)

    #     x_center = self.x_center[index]
    #     y_center = self.y_center[index]

    #     # If z is fixed, we only sample x_src,y_src => 2D problem
    #     ndim = 2

    #     # function to pick random initial positions within prior
    #     def random_in_prior():
    #         x0 = np.random.uniform(x_center - x_range, x_center + x_range)
    #         y0 = np.random.uniform(y_center - y_range, y_center + y_range)
    #         return np.array([x0, y0])

    #     initial_positions = np.array([random_in_prior() for _ in range(n_walkers)])

    #     with mp.Pool(processes=n_processes) as pool:
    #         sampler = emcee.EnsembleSampler(
    #             n_walkers,
    #             ndim,
    #             _log_posterior_func,
    #             # pass arguments in the exact order of the function signature after 'params'
    #             args=(x_center, y_center, x_range, y_range,
    #                   self,             # <-- self_obj
    #                   dt_true, index, sigma,
    #                   fix_z, z_s_fix,   # <--- controlling the 'fix_z' logic
    #                   0, 100),          # z_lower=0, z_upper=100 (if you want them changed, do so here)
    #             pool=pool
    #         )
    #         sampler.run_mcmc(initial_positions, n_steps, progress=True)

    #     chain = sampler.get_chain(discard=burn_in, flat=False)
    #     flat_samples = chain.reshape((-1, ndim))
    #     return sampler, flat_samples

    # def localize_mcmc_emcee(self, dt_true, index=0,
    #                     n_walkers=50, n_steps=5000, burn_in=1000,
    #                     x_range=50.0, y_range=50.0,
    #                     z_lower=2.5, z_upper=3.5,
    #                     sigma=0.05,
    #                     random_seed=42,
    #                     n_processes=8):
    #     """
    #     MCMC with parallelization via emcee + multiprocessing,
    #     sampling (x_src, y_src, z_s) within:
    #     x_src in [x_center - x_range, x_center + x_range]
    #     y_src in [y_center - y_range, y_center + y_range]
    #     z_s   in [z_lower, z_upper].

    #     The log-posterior function `_log_posterior_func` is:
    #         def _log_posterior_func(params,
    #                                 x_center, y_center, x_range, y_range,
    #                                 self_obj,
    #                                 dt_true, index, sigma,
    #                                 fix_z, z_s_fix,
    #                                 z_lower=0, z_upper=100):
    #             ...
    #     Hence we pass arguments in the same order after 'params'.
    #     """
    #     np.random.seed(random_seed)

    #     # 1) Grab the lens center for this index
    #     x_center = self.x_center[index]
    #     y_center = self.y_center[index]

    #     # 2) 3D parameter space => (x_src, y_src, z_s)
    #     ndim = 3

    #     # Helper: random positions in the bounding box
    #     def random_in_prior():
    #         x0 = np.random.uniform(x_center - x_range, x_center + x_range)
    #         y0 = np.random.uniform(y_center - y_range, y_center + y_range)
    #         z0 = np.random.uniform(z_lower, z_upper)
    #         return np.array([x0, y0, z0])

    #     # Create initial positions for all walkers
    #     initial_positions = np.array([random_in_prior() for _ in range(n_walkers)])

    #     # Define a custom stretch move (optional)
    #     move = StretchMove(a=1.8)

    #     # 3) Parallel pool
    #     with mp.Pool(processes=n_processes) as pool:
    #         sampler = emcee.EnsembleSampler(
    #             nwalkers=n_walkers,
    #             ndim=ndim,
    #             log_prob_fn=_log_posterior_func,
    #             # Match the signature:
    #             #   _log_posterior_func(params,
    #             #       x_center, y_center, x_range, y_range,
    #             #       self_obj,
    #             #       dt_true, index, sigma,
    #             #       fix_z, z_s_fix,
    #             #       z_lower=0, z_upper=100)
    #             args=(
    #                 x_center, y_center, x_range, y_range,
    #                 self,               # <-- self_obj
    #                 dt_true, index, sigma,
    #                 False, None,        # fix_z=False, z_s_fix=None
    #                 z_lower, z_upper    # override the defaults in the function
    #             ),
    #             pool=pool,
    #             moves=move
    #         )

    #         # 4) Run MCMC
    #         sampler.run_mcmc(initial_positions, n_steps, progress=True)

    #     # 5) Discard burn-in, flatten
    #     chain = sampler.get_chain(discard=burn_in, flat=False)
    #     flat_samples = chain.reshape((-1, ndim))

    #     return sampler, flat_samples

    def localize_diffevo_then_mcmc_known_cluster(self, dt_true, index=0,
                               # DE settings
                               early_stop=1e6,
                               # MCMC settings
                               n_walkers=24, n_steps=800, burn_in=300,
                               x_range_prior=10.0, y_range_prior=10.0,
                               x_range_int=2.0, y_range_int = 2.0, z_range_int = 0.3,
                               z_lower=2.0, z_upper=3.5,
                               sigma=0.05,
                               n_processes=8):
        """
        1) Use differential evolution to find (x_opt, y_opt, z_opt).
        2) Then run MCMC around that solution to get posterior samples.
        """

        # -------------------------------------------------------
        # STEP 1: Run differential evolution to get best guess
        # -------------------------------------------------------
        # You can adapt localize_known_cluster_diffevo_with_z to use the 
        # de_xrange, de_yrange, de_zrange if needed. For now let's just call it directly.
        x_opt, y_opt, z_opt, min_chi_sq = self.localize_known_cluster_diffevo_with_z(dt_true, index, threshold=early_stop)
        if x_opt is None and y_opt is None:
            print("DE failed, try another cluster.")
            return None, None, None, None
        
        print(f"DE best solution for cluster {index}: x={x_opt:.3f}, y={y_opt:.3f}, z={z_opt:.3f}, chi^2={min_chi_sq:.3f}")

        x_center = x_opt
        y_center = y_opt

        # -------------------------------------------------------
        # STEP 2: MCMC around that solution
        # -------------------------------------------------------
        # We'll do a 3D MCMC on (x_src, y_src, z_s).
        # We'll define a bounding box for the prior around the DE solution,
        # e.g. ± x_range for x, ± y_range for y, and [z_lower, z_upper] for z.

        # 1) Create initial positions near the DE solution
        ndim = 3
        def random_in_prior_around_de():
            x0 = np.random.uniform(x_opt - x_range_int, x_opt + x_range_int)
            y0 = np.random.uniform(y_opt - y_range_int, y_opt + y_range_int)
            z0 = np.random.uniform(z_opt - z_range_int, z_opt + z_range_int)
            if z0 < z_lower:
                z0 = z_lower
            if z0 > z_upper:
                z0 = z_upper
            return np.array([x0, y0, z0])

        initial_positions = np.array([random_in_prior_around_de() for _ in range(n_walkers)])

        # 2) Larger a, more aggressive stretch move
        move = emcee.moves.StretchMove(a=1.9)

        with mp.Pool(processes=n_processes) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers=n_walkers,
                ndim=ndim,
                log_prob_fn=_log_posterior_func,
                args=(
                    x_center, y_center, x_range_prior, y_range_prior,
                    self,               # <-- self_obj
                    dt_true, index, sigma,
                    False, None,        # fix_z=False, z_s_fix=None
                    z_lower, z_upper    # override the defaults in the function
                    ),    
                pool=pool,
                moves=move
            )

            # 3) Run MCMC
            sampler.run_mcmc(initial_positions, n_steps, progress=True)

        # 4) Discard burn-in, flatten
        flat_samples = sampler.get_chain(discard=burn_in, flat=True)
        #flat_samples = chain.reshape((-1, ndim))

        # 5) Analyze or return results
        # e.g. median or best fit from MCMC
        x_median = np.median(flat_samples[:,0])
        y_median = np.median(flat_samples[:,1])
        z_median = np.median(flat_samples[:,2])
        print(f"MCMC median after DE: x={x_median:.2f}, y={y_median:.2f}, z={z_median:.2f}")

        return (x_opt, y_opt, z_opt, min_chi_sq), (x_median, y_median, z_median), sampler, flat_samples
    
    def localize_diffevo_then_mcmc(self, dt_true,
                               # DE settings
                               early_stop=1e6,
                               # MCMC settings
                               n_walkers=24, n_steps=800, burn_in=300,
                               x_range_prior=10.0, y_range_prior=10.0,
                               x_range_int=1.0, y_range_int = 1.0, z_range_int = 0.2,
                               z_lower=2.0, z_upper=3.5,
                               sigma=0.10,
                               n_processes=8):

        opt_pos = None
        opt_chi_sq = None
        opt_sampler = None
        opt_flat_samples = None
        opt_index = None
        

        for i in range(6):
            index = i
            _, medians, sampler, flat_samples = self.localize_diffevo_then_mcmc_known_cluster(dt_true,  index,
                               early_stop,
                               n_walkers, n_steps, burn_in,
                               x_range_prior, y_range_prior,
                               x_range_int, y_range_int, z_range_int,
                               z_lower, z_upper,
                               sigma,
                               n_processes)
            
            if medians is None:
                print("Nothing found for this index.")
                continue

            log_probs = sampler.get_log_prob(discard=burn_in, flat=True)  # shape (n_samples,)
            # 3) Find the index of the maximum log-likelihood
            best_idx = np.argmax(log_probs)

            # 4) Extract the parameter set with largest log-likelihood
            best_params = flat_samples[best_idx]
            chi_sq = self.chi_squared_with_z(best_params, dt_true, index)
            
            if opt_chi_sq is None or chi_sq <= opt_chi_sq:
                opt_pos = best_params
                opt_chi_sq = chi_sq
                opt_sampler = sampler
                opt_flat_samples = flat_samples
                opt_index = index
                print("Replaced original opt.")

        return opt_index, opt_pos, opt_chi_sq, opt_sampler, opt_flat_samples

    def chi_squared_vector(self, src_guesses, dt_true, index=0, sigma=0.05):
        chi_sqs = np.zeros(src_guesses.shape[0])
        for i, src_guess in enumerate(src_guesses):
            x_src, y_src = src_guess
            img = self.image_position(x_src, y_src, index)
            if not img or len(img[0]) != len(dt_true):
                chi_sqs[i] = 1e6
                continue
            t = self.time_delay(img[0], img[1], index, x_src, y_src)
            dt = [ti - min(t) for ti in t]
            chi_sq = sum(((dt_i - dt_true_i) / sigma) ** 2 for dt_i, dt_true_i in zip(dt, dt_true))
            chi_sqs[i] = np.log1p(chi_sq)
        return chi_sqs
