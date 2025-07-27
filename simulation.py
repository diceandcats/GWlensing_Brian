# pylint: skip-file
import numpy as np
from lensing_data_class import LensingData
from cluster_local_tidy import ClusterLensing
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import pandas as pd
import corner
import arviz as az
import pathlib

import warnings

# Suppressing the lenstronomy warning on astropy.cosmology
from lenstronomy.LensModel.lens_model import LensModel
warnings.filterwarnings("ignore", category=UserWarning, module='lenstronomy.LensModel.lens_model')

# inject numerical lens model

scenarios = {
        '1': 'abell370',
        '2': 'abell2744',
        '3': 'abells1063',
        '4': 'macs0416',
        '5': 'macs0717',
        '6': 'macs1149'
    }

full_cluster_names = {
        'abell370': 'Abell 370',
        'abell2744': 'Abell 2744',
        'abells1063': 'Abell S1063',
        'macs0416': 'MACS J0416.1-2403',
        'macs0717': 'MACS J0717.5+3745',
        'macs1149': 'MACS J1149.5+2223'
    }

# Initialize lists to store the data arrays
datax_list = []
datay_list = []
data_psi_list = []

for i in scenarios:
    clustername = scenarios[i]
    full_cluster_name = full_cluster_names[clustername]

    file_dir = os.getcwd()
    fits_filex = os.path.join(
        file_dir,
        f'Research/GWlensing_Brian/GCdata/{full_cluster_name}/diego/hlsp_frontier_model_{clustername}_diego_v4.1_x-arcsec-deflect.fits'
    )
    fits_filey = os.path.join(
        file_dir,
        f'Research/GWlensing_Brian/GCdata/{full_cluster_name}/diego/hlsp_frontier_model_{clustername}_diego_v4.1_y-arcsec-deflect.fits'
    )
    psi_file = os.path.join(
        file_dir,
        f'Research/GWlensing_Brian/GCdata/{full_cluster_name}/diego/hlsp_frontier_model_{clustername}_diego_v4.1_psi.fits'
    )

    with fits.open(fits_filex) as hdulx, fits.open(fits_filey) as hduly, fits.open(psi_file) as hdul_psi:
        datax = hdulx[0].data
        datay = hduly[0].data
        data_psi = hdul_psi[0].data

        # Append the data arrays to the lists
        datax_list.append(datax)
        datay_list.append(datay)
        data_psi_list.append(data_psi)

# initialize the LensingData object
pixscale_list = [0.42, 0.51, 0.51, 0.42, 0.42, 0.42]
lensing_data = LensingData(
    alpha_maps_x=datax_list,
    alpha_maps_y=datay_list,
    lens_potential_maps=data_psi_list,
    pixscale = pixscale_list,
    z_l_list = [0.375, 0.308, 0.351, 0.397, 0.545, 0.543],
)

#Initialize the Main Analysis Class
z_s_ref = 1.5  # Reference source redshift
cluster_system = ClusterLensing(data=lensing_data, z_s_ref=z_s_ref)

print("Setup complete. Lensing system initialized.")

# get the dt_true
# real image pos and dt
real_params = {"x_src" : 96.3642998642583, "y_src": 96.24579242317353, "z_s": 3.668910475026701, "H0": 75.60819770277777}
real_cluster = 0
# Calculate the image positions and time delays for the test parameters
output = cluster_system.calculate_images_and_delays(
    real_params, real_cluster
)
print(f'True time delays: {output['time_delays']}')
dt_true = output['time_delays']

# luminosity distance calculation
cosmos = FlatLambdaCDM(H0=real_params['H0'], Om0=0.3)
lum_dist_true = cosmos.luminosity_distance(real_params['z_s']).value  # Luminosity distance in Mpc
print("True luminosity distances:", lum_dist_true)

# Define the output directory for results
base_output_dir = pathlib.Path("/home/dices/Research/GWlensing_Brian/oddratio")
base_output_dir.mkdir(exist_ok=True)  # Ensure the parent directory exists
max_n = 0
for path in base_output_dir.glob('test_tidy_*'):
    if path.is_dir():
        # Extract the number from the directory name 'test_tidy_n'
        n = int(path.name.split('_')[-1])
        if n > max_n:
            max_n = n

# Create the new directory with n+1
new_dir_name = f"test_tidy_{max_n + 1}"
OUT_DIR = base_output_dir / new_dir_name
OUT_DIR.mkdir(exist_ok=True)

# test de then mcmc 
print("\nRunning the full analysis pipeline...")
# # Define settings for the MCMC sampler
# mcmc_settings = {
#     "n_walkers": 20,      # Number of MCMC walkers
#     "n_steps": 8000,      # Number of steps per walker
#     "fit_z": True,        # We want to fit for the source redshift (z_s)
#     "fit_hubble": True,   # We want to fit for the Hubble constant (H0)
#     "lum_dist_true": lum_dist_true, # An external "true" luminosity distance measurement (in Mpc)
#     "sigma_lum": 0.05,     # The fractional error on the luminosity distance
#     # Define the prior boundaries for the parameters being fit in the MCMC
#     "z_bounds": (1.0, 5.0),
#     "H0_bounds": (60, 100)
# }

# # Call the main analysis function
# # The function will first run DE to find the best cluster,
# # then run MCMC on that best-fit model.
# mcmc_results, accepted_cluster_indices = cluster_system.find_best_fit(
#     dt_true=dt_true,
#     run_mcmc=True,
#     mcmc_settings=mcmc_settings
# )
# print(f"\nAnalysis complete. Found {len(mcmc_results)} cluster(s) meeting the MCMC criterion.")