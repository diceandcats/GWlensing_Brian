import numpy as np
#import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import pyswarms as ps
#from pyswarms.utils.functions import single_obj as fx
from astropy.cosmology import FlatLambdaCDM
#import lenstronomy.Util.constants as const
import pandas as pd
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from tqdm import tqdm

# pylint: disable=C0103
class ClusterLensing_fyp:
    """
    Class for localization
    """

    def __init__(self, alpha_maps_x, alpha_maps_y, lens_potential_maps, z_l,
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
        self.z_l = z_l
        self.z_s = z_s
        self.pixscale = pixscale
        self.image_positions = None
        self.magnifications = None
        self.time_delays = None
        self.diff_z = diff_z
        self.scale_factor = None #for scaling the deflection maps

        self.x_center, self.y_center = [90,75,110,70,90,70], [70,80,95,60,93,65]

        # get the size of the deflection maps
        self.size = []
        for i in range(6):
            self.size.append(len(alpha_maps_x[i]))

        self.lens_models = []
        self.solvers = []
        self.kwargs_list = []
        for i in range(6):

            x_grid = np.linspace(0, self.size[i] - 1, self.size[i]) * pixscale[i]
            kwargs = {
                'grid_interp_x': x_grid,
                'grid_interp_y': x_grid,
                'f_': lens_potential_maps[i]*pixscale[i]**2,
                'f_x': alpha_maps_x[i],
                'f_y': alpha_maps_y[i]
            }
            self.kwargs_list.append(kwargs)
            lens_model_list = ['INTERPOL']
            lensmodel = LensModel(lens_model_list=lens_model_list, z_source=z_s, z_lens=z_l)
            solver = LensEquationSolver(lensmodel)
            self.lens_models.append(lensmodel)
            self.solvers.append(solver)

        if diff_z:
            self.D_S1, self.D_S2, self.D_LS1, self.D_LS2 = self.scaling()

    def scaling(self):
        """
        Scale the deflection and lens potential maps.
        """
        # Redshifts
        z_L = self.z_l
        z_S = self.z_s

        # Calculate distances
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        D_S1 = cosmo.angular_diameter_distance(1.0)
        D_S2 = cosmo.angular_diameter_distance(z_S)
        D_LS1 = cosmo.angular_diameter_distance_z1z2(z_L, 1.0)
        D_LS2 = cosmo.angular_diameter_distance_z1z2(z_L, z_S)

        # Scale factor
        scal = (D_LS1 * D_S2) / (D_LS2 * D_S1)

        # Scale deflection maps
        for i in range(6):
            self.alpha_maps_x[i] *= scal
            self.alpha_maps_y[i] *= scal

        return D_S1, D_S2, D_LS1, D_LS2
    
    

    def image_position(self, x_src, y_src, index=0):
        #lens_model = self.lens_models[index]
        solver = self.solvers[index]
        kwargs_lens = [self.kwargs_list[index]]
        image_positions = solver.image_position_from_source(
            x_src, y_src, kwargs_lens,  # Empty kwargs_lens since grids are set in LensModel
            min_distance=self.pixscale[index],
            search_window=100,
            verbose=False,
            x_center=self.x_center[int(index)],
            y_center=self.y_center[int(index)]
        )
        return image_positions

    def image_position_z(self, x_src, y_src, z, index=0):
        lens_model_z = LensModel(lens_model_list=['INTERPOL'], z_source=z, z_lens=self.z_l)
        solver_z = LensEquationSolver(lens_model_z)
        kwargs_lens = [self.kwargs_list[index]]
        image_positions = solver_z.image_position_from_source(
            x_src, y_src, kwargs_lens,
            min_distance=self.pixscale[index],
            search_window=100,
            verbose=False,
            x_center=self.x_center[int(index)],
            y_center=self.y_center[int(index)]
        )
        return image_positions
    
    def rand_src_test(self, n_each=10, n=5):
        """
        Generate random source positions with over 'n' images for each cluster.

        Parameters:
        -----------
        n_each: int
            Number of random source positions to generate per cluster.
        n: int
            Minimum number of images required for a source position to be accepted.

        Returns:
        --------
        srcs: list
            List of accepted source positions.
        indices: list
            List of cluster indices corresponding to the accepted source positions.
        """
        srcs = []
        indices = []
        clusters = 6  # Number of clusters

        for cluster_index in range(clusters):
            count = 0
            while count < n_each:
                # Generate random source position
                x_center = self.x_center[int(cluster_index)]
                x_src = np.random.uniform(x_center - 30, x_center + 30)  # Adjust the range as needed
                y_center = self.y_center[int(cluster_index)]
                y_src = np.random.uniform(y_center - 30, y_center + 30)  # Adjust the range as needed

                # Get image positions
                img_positions = self.image_position(x_src, y_src, cluster_index)

                # Debugging: Print the number of images
                
                # Check if the number of images is over the threshold
                if len(img_positions[0]) >= n:
                    print(f"Cluster index: {cluster_index}, Number of images: {len(img_positions[0])}")
                    srcs.append((x_src, y_src))
                    indices.append(cluster_index)
                    count += 1  # Increment count for the current cluster

        return srcs, indices
    

    def rand_src_test_1_cluster(self, n_each=50, n=5, index = 0):
        """
        Generate random source positions with over 'n' images for each cluster.

        Parameters:
        -----------
        n_each: int
            Number of random source positions to generate per cluster.
        n: int
            Minimum number of images required for a source position to be accepted.

        Returns:
        --------
        srcs: list
            List of accepted source positions.
        indices: list
            List of cluster indices corresponding to the accepted source positions.
        """
        srcs = []
        indices = []

        
        count = 0
        while count < n_each:
            # Generate random source position
            x_center = self.x_center[int(index)]
            x_src = np.random.uniform(x_center - 30, x_center + 30)  # Adjust the range as needed
            y_center = self.y_center[int(index)]
            y_src = np.random.uniform(y_center - 30, y_center + 30)  # Adjust the range as needed

            # Get image positions
            img_positions = self.image_position(x_src, y_src, index)

            # Debugging: Print the number of images
                
            # Check if the number of images is over the threshold
            if len(img_positions[0]) >= n:
                print(f"Cluster index: {index}, Number of images: {len(img_positions[0])}")
                srcs.append((x_src, y_src))
                indices.append(index)
                count += 1  # Increment count for the current cluster

        return srcs, indices
    
    def correct_indices_first(self, df):
        corrected_indices = []

        for idx, row in tqdm(df.iterrows()):
            x = row['x']
            y = row['y']
            indices = row['indices']
            indices = int(indices)
            image_positions = self.image_position(x, y, indices)
            no_images = len(image_positions[0])
            success = False
            if no_images < 5:
                for possible_index in range(6):
                    image_positions = self.image_position(x, y, possible_index)
                    length_indices = len(image_positions[0])
                    if length_indices >= 5:
                        corrected_indices.append(possible_index)
                        success = True
                        break
                if not success:
                    print(f'error in row {idx}')
                    corrected_indices.append(indices)
            else:
                corrected_indices.append(indices)

        # Update the DataFrame
        df['indices'] = corrected_indices

        return df

    
    def time_delay(self,x_img, y_img, index=0, x_src=None, y_src=None):
        """
        Find the time delay between the source and image positions.

        Parameters:
        -----------
        x_src: float
            The x coordinate of the source in arcsec.
        y_src: float
            The y coordinate of the source in arcsec.
        x_img: float
            The x coordinate of the image in arcsec.
        y_img: float
            The y coordinate of the image in arcsec.
        index: int
            Index of the deflection map set to use (0 to 5).
        """
        kwargs = self.kwargs_list[index]
        lens_model = self.lens_models[index]
        t = lens_model.arrival_time(x_img, y_img, [kwargs], x_source=x_src, y_source=y_src)
        dt= t - t.min()
        return dt

    def chi_squared(self, src_guess, dt_true, index=0, sigma=0.05):
        """
        Calculate the chi-squared value for a given source position guess.
        """
        x_src, y_src = src_guess
        img = self.image_position(x_src, y_src, index)
        
        # Check if image positions are found
        if len(img[0]) == 0:
            # No images found; return a high chi-squared penalty
            return 1e13
        
        # if the lengths are not equal, set the penalty
        len_dt_true = len(dt_true)
        img_no = len(img[0])
        if img_no != len_dt_true:
            return abs(img_no - len_dt_true) * 1.4e12  # Penalty value

        t = self.time_delay(img[0], img[1], index, x_src, y_src)
        dt = t-t.min()

        # Calculate chi-squared using NumPy vectorization
        chi_sq = np.sum((dt - dt_true) ** 2) / (2 * sigma ** 2)
        return chi_sq
    
    def chi_squared_with_z(self, src_guess, z, dt_true, index=0, sigma=0.05):
        """
        Calculate the chi-squared value for a given source position guess with redshift scaling.
        """
        x_src, y_src = src_guess
        img = self.image_position_z(x_src, y_src, z, index)
        
        # Check if image positions are found
        if len(img[0]) == 0:
            # No images found; return a high chi-squared penalty
            return 1e13
        
        # if the lengths are not equal, set the penalty
        len_dt_true = len(dt_true)
        img_no = len(img[0])
        if img_no != len_dt_true:
            return abs(img_no - len_dt_true) * 1.4e12  # Penalty value

        t = self.time_delay(img[0], img[1], index, x_src, y_src)
        dt = t-t.min()

        # Calculate chi-squared using NumPy vectorization
        chi_sq = np.sum((dt - dt_true) ** 2) / (2 * sigma ** 2)
        return chi_sq
    
    def localize_known_cluster(self, x_src_guess, y_src_guess, dt_true, index=1):
        """
        Find the source position by minimizing the chi-squared value 
        and find which index of cluster does the source located in.

        Parameters:
        -----------
        x_img: list
            List of x coordinates of the image in arcsec.
        y_img: list
            List of y coordinates of the image in arcsec.
        x_src: float
            The x coordinate of the source in arcsec.
        y_src: float
            The y coordinate of the source in arcsec.
        index: int
            Index of the deflection map set to use (0 to 5).
        """
        i = index
        src_guess = [x_src_guess, y_src_guess]
        result = minimize(self.chi_squared, src_guess, args=(dt_true, i),method='L-BFGS-B',
            tol=1e-7)

        min_chi_sq = result.fun
        return result.x[0], result.x[1], min_chi_sq

    
    def localize_known_cluster_diffevo(self, dt_true, index=1):
        """
        Find the source position by minimizing the chi-squared value
        using differential evolution.
        """
        x_center = self.x_center[int(index)]
        y_center = self.y_center[int(index)]
        x_min, x_max = x_center - 50, x_center + 50
        y_min, y_max = y_center - 50, y_center + 50
        bounds = [(x_min, x_max), (y_min, y_max)]  # Define appropriate bounds

        # Define the callback function to stop optimization
        def callback_fn(xk, convergence):
            # Compute the chi-squared value at the current parameters
            func_value = self.chi_squared(xk, dt_true, index)
            # If chi-squared is less than 1e-5, stop the optimization
            if func_value < 1e-4:
                return True  # Stops the optimization
            else:
                return False
            
        result = differential_evolution(
            self.chi_squared,  # Use the transformed objective function
            bounds,
            args=(dt_true, index),
            strategy='rand1bin',
            maxiter=250,           # Decreased iterations
            popsize=40,             # Larger population size
            tol=1e-3,               # Larger tolerance for faster running time
            mutation=(0.5, 1),
            recombination=0.7,
            polish=False,
            updating='deferred',    # May improve performance
            workers=-1,
            disp=False,
            callback=callback_fn)
            

        x_opt, y_opt = result.x
        min_chi_sq = result.fun
        return x_opt, y_opt, min_chi_sq
    

    def localize_known_cluster_diffevo_with_z(self, dt_true, index=1):
        """
        Find the source position (x, y) and redshift z by minimizing the chi-squared
        value using differential evolution.
        """
        # Center points for x and y (use your existing logic)
        x_center = self.x_center[int(index)]
        y_center = self.y_center[int(index)]
        
        # Define search bounds for x, y, and z
        x_min, x_max = x_center - 50, x_center + 50
        y_min, y_max = y_center - 50, y_center + 50
        
        # Example z-bounds
        z_min, z_max = 0.51, 1.1
        
        # Combine all bounds (x, y, z)
        bounds = [(x_min, x_max),
                (y_min, y_max),
                (z_min, z_max)]
        
        # Define the callback function to potentially stop optimization early
        def callback_fn(xk, convergence):
            # Compute the chi-squared value at the current parameters
            func_value = self.chi_squared_with_z(xk, dt_true, index)
            # If chi-squared is less than 1e-5, stop the optimization
            if func_value < 1e-4:
                return True  # Stops the optimization
            else:
                return False

        # Perform the differential evolution
        result = differential_evolution(
            self.chi_squared_with_z,   # Objective function with z included
            bounds,
            args=(dt_true, index),     # Additional arguments passed to the objective
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
            callback=callback_fn,

        )
        
        # Unpack optimized x, y, z
        x_opt, y_opt, z_opt = result.x
        min_chi_sq = result.fun

        return x_opt, y_opt, z_opt, min_chi_sq

    
    def localize_diffevo(self, dt_true):
        """
        Find the source position by minimizing the chi-squared value
        using differential evolution.
        """
        chi_sqs = []
        src_guess = []
        for i in range(6):
            x_guess, y_guess, chi_sq = self.localize_known_cluster_diffevo(dt_true, i)
            src_guess.append([x_guess, y_guess])
            chi_sqs.append(chi_sq)
        min_chi_sq = min(chi_sqs)
        index = chi_sqs.index(min_chi_sq)
        return index, src_guess[index], min_chi_sq, src_guess, chi_sqs
        
    
    def chi_squared_vector(self, src_guesses, dt_true, index=0, sigma=0.05):
        chi_sqs = np.zeros(src_guesses.shape[0])  # Initialize array to hold chi-squared values

        for i, src_guess in enumerate(src_guesses):
            x_src, y_src = src_guess
            # Compute image positions
            img = self.image_position(x_src, y_src, index)
            if not img or len(img[0]) != len(dt_true):
                # Assign a penalty value to chi_sqs[i]
                chi_sqs[i] = 1e6  # Penalty value
                continue  # Skip to next iteration

            # Compute time delays
            t = self.time_delay(img[0], img[1], index, x_src, y_src)
            dt = [ti - min(t) for ti in t]

            # Compute chi-squared
            chi_sq = sum(((dt_i - dt_true_i) / sigma) ** 2 for dt_i, dt_true_i in zip(dt, dt_true))

            # Apply logarithmic scaling
            chi_sqs[i] = np.log1p(chi_sq)

        return chi_sqs


    def localize_known_cluster_pso(self, dt_true, index=1, n_particles=50, iters=100):
        """
        Use PSO to localize the source position for a known cluster index.
        """
        # Define bounds for the source coordinates based on the cluster center
        x_center = self.x_center[int(index)]
        y_center = self.y_center[int(index)]
        x_min, x_max = x_center - 50, x_center + 50
        y_min, y_max = y_center - 50, y_center + 50
        bounds = ([x_min, y_min], [x_max, y_max])
        
        # Define the number of dimensions (2 for x_src and y_src)
        dimensions = 2
        
        # Define options for the PSO algorithm with dynamic inertia weight
        options = {
            'c1': 1.5,  # Cognitive parameter
            'c2': 1.5,  # Social parameter
            'w': 0.7    # Starting inertia weight
        }

        # Generate initial positions near the initial guess
        init_pos = np.random.uniform(
            low=[x_center - 3, y_center - 3],
            high=[x_center + 3, y_center + 3],
            size=(n_particles, dimensions)
        )
        # Ensure initial positions are within bounds
        init_pos = np.clip(init_pos, a_min=bounds[0], a_max=bounds[1])
       
        # Create the optimizer
        optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=dimensions,
            options=options,
            bounds=bounds,
            init_pos=init_pos,
            velocity_clamp=(-10, 10)
            )
        
        # Define the objective function wrapper
        def objective_function(src_guesses):
            return self.chi_squared_vector(src_guesses, dt_true, index)
        
        # Perform the optimization
        best_cost, best_pos = optimizer.optimize(
            objective_function,
            iters=iters,
            verbose=True  # Set to True to see progress
        )
        
        x_opt, y_opt = best_pos
        min_chi_sq = best_cost
        return x_opt, y_opt, min_chi_sq
    
    