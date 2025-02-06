import numpy as np
from scipy.optimize import minimize, differential_evolution
import pyswarms as ps
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

# pylint: disable=C0103
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
            search_window=100,
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
            search_window=100,
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
            return 1e13
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
            return 1e13
        if len(img[0]) != len(dt_true):
            return abs(len(img[0]) - len(dt_true)) * 1.4e12

        candidate_lens_model = LensModel(lens_model_list=['INTERPOL'], z_source=z_s, z_lens=self.z_l_list[index])
        t = candidate_lens_model.arrival_time(img[0], img[1], [candidate_kwargs],
                                               x_source=x_src, y_source=y_src)
        dt_candidate = t - t.min()
        chi_sq = np.sum((dt_candidate - dt_true)**2) / (2 * sigma**2)
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
            tol=1e-3,
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

    def localize_known_cluster_diffevo_with_z(self, dt_true, index=1):
        x_center = self.x_center[int(index)]
        y_center = self.y_center[int(index)]
        x_min, x_max = x_center - 50, x_center + 50
        y_min, y_max = y_center - 50, y_center + 50
        z_lower = 2.5
        z_upper = 3.5
        bounds = [(x_min, x_max), (y_min, y_max), (z_lower, z_upper)]

        # def objective(params):
        #     x_src, y_src, z_candidate = params
        #     return self.chi_squared_with_z((x_src, y_src), z_candidate, dt_true, index=index, sigma=sigma)

        def callback_fn(xk, convergence):
            if self.chi_squared_with_z(xk, dt_true, index) < 1e-2:
                return True
            return False

        result = differential_evolution(
            self.chi_squared_with_z,
            bounds,
            args=(dt_true, index),
            strategy='rand1bin',
            maxiter=250,
            popsize=40,
            tol=1e-3,
            mutation=(0.5, 1),
            recombination=0.7,
            polish=False,
            updating='deferred',
            workers=-1,
            disp=False,
            callback=callback_fn
            )
        
        x_opt, y_opt, z_opt = result.x
        min_chi_sq = result.fun
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
