import numpy as np
from astropy.io import fits
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
# for psi uncertainty maps
Name = "Abell 2744"
name = "abell2744"
data_path = f"/home/dices/Research/GWlensing_Brian/GCdata/{Name}/cats/range/"  # change this to the actual path where your FITS files are located
#Build the file pattern
psi_file_pattern = os.path.join(data_path, f"hlsp_frontier_model_{name}_cats-map???_v4.1_psi.fits")
file_list_psi = sorted(glob.glob(psi_file_pattern))
N_files_psi = len(file_list_psi)
print(f"Found {N_files_psi} psi FITS files.")

file_list = file_list_psi
N_files = len(file_list)

# Read the first FITS to get dimensions
first_map = fits.getdata(file_list[0])
H, W = first_map.shape
print(f"Dimensions of the potential maps: {H} x {W}")
# Pre-allocate an array for the potentials: shape (N_files, H, W)
potentials = np.empty((N_files, H, W))

for i, filename in enumerate(file_list):
    data = fits.getdata(filename)
    if data.shape != (H, W):
        raise ValueError(f"File {filename} has shape {data.shape}, expected ({H}, {W})")
    potentials[i] = data

# pixel size (arcsec/pixel)
dpsi=0.3

# Load psi maps
psi_maps = np.empty((N_files_psi, H, W), dtype=np.float64)
for i in range(N_files_psi):
    psi_maps[i] = fits.getdata(file_list_psi[i]).astype(np.float64)

# Ensemble mean of components
psi_mean = np.mean(psi_maps, axis=0)                # Shape: (H, W)

# Take sample std across the ensemble for each pixel.
# This is the corrected line.
sigma_abs = np.std(psi_maps, axis=0, ddof=1)           # Shape: (H, W)

# Relative uncertainty map (per pixel)
eps = 1e-12     # Avoid division by zero
relative_sigma = sigma_abs / np.maximum(psi_mean, eps)

# Overall upper bound (84th percentile of the relative uncertainties)
upper_bound_fraction = np.percentile(relative_sigma, 84.0)
upper_bound_percentage = upper_bound_fraction * 100.0
print(f"Upper bound relative uncertainty in |ψ|: {upper_bound_percentage:.4f}%")

