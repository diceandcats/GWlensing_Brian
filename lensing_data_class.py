from dataclasses import dataclass, field
from typing import List
import numpy as np

@dataclass
class LensingData:
    """
    A data class to store and manage the input data for cluster lensing analysis.

    Attributes:
    -----------
    alpha_maps_x : List[np.ndarray]
        List of deflection maps in the x-direction for each cluster (arcsec).
    alpha_maps_y : List[np.ndarray]
        List of deflection maps in the y-direction for each cluster (arcsec).
    lens_potential_maps : List[np.ndarray]
        List of lens potential maps for each cluster (arcsec^2).
    pixscale : List[float]
        List of pixel scales for each cluster's maps.
    z_l_list : List[float]
        List of lens redshifts for each cluster.
    x_center : List[float]
        List of x-coordinates for the center of the search window for each cluster.
    y_center : List[float]
        List of y-coordinates for the center of the search window for each cluster.
    search_window_list : List[float]
        List of search window sizes for each cluster.
    """
    alpha_maps_x: List[np.ndarray]
    alpha_maps_y: List[np.ndarray]
    lens_potential_maps: List[np.ndarray]
    pixscale: List[float]
    z_l_list: List[float]
    
    # Default cluster-specific parameters from the original code
    x_center: List[float] = field(default_factory=lambda: [99, 132, 131, 109, 125, 107.1])
    y_center: List[float] = field(default_factory=lambda: [95, 100, 131, 111, 112, 113.1])
    search_window_list: List[float] = field(default_factory=lambda: [100, 95, 95, 100, 120, 100])
    uncertainty_alpha: List[float] = field(default_factory=lambda: [0.043573, 0.086522, 0.068417, 0.061110, 0.021551, 0.084857])