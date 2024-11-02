from math import ceil, floor
import os
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import scipy.optimize._minimize as minimize
from astropy.cosmology import FlatLambdaCDM
import lenstronomy.Util.constants as const
import pandas as pd
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

# pylint: disable=C0103
class ClusterLensing:
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
        pixscale: The pixel scale.
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

        # get the size of the deflection maps
        self.size = []
        for i in range(6):
            self.size.append(len(alpha_maps_x[i]))

        lens_model = ['INTERPOL']
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
        self.lensmodel = LensModel(lens_model_list=lens_model, z_source=z_s, z_lens=z_l)
        self.solver = LensEquationSolver(self.lensmodel)

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
        """
        Find the image positions of the source for a given deflection map set.

        Parameters:
        -----------
        x_src: float
            The x coordinate of the source in arcsec.
        y_src: float
            The y coordinate of the source in arcsec.
        index: int
            Index of the deflection map set to use (0 to 5).
        """
        kwargs = self.kwargs_list[index]
        size = self.size[index]
        search_start = size/2*self.pixscale[index]
        image_positions = self.solver.image_position_from_source(
            x_src, y_src, [kwargs], min_distance=self.pixscale[index],
            search_window=100, verbose=False, x_center=search_start, y_center=search_start)
        return image_positions
    
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
        t = self.lensmodel.arrival_time(x_img, y_img, [kwargs], x_source=x_src, y_source=y_src)
        dt = []
        for i in range(len(x_img)):
            dt.append(t[i] - min(t))
        return dt
