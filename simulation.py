# pylint: skip-file
import numpy as np
from lensing_data_class import LensingData
from cluster_local_tidy import ClusterLensing
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import pandas as pd
import corner
import arviz as az
import pathlib
import argparse
import warnings

from csv_lock import update_csv_row

# Suppressing the lenstronomy warning on astropy.cosmology
from lenstronomy.LensModel.lens_model import LensModel
warnings.filterwarnings("ignore", category=UserWarning, module='lenstronomy.LensModel.lens_model')


# Parse CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True, help="Path to input CSV (with indices,x,y,z,H0,...)")
parser.add_argument("--row", type=int, required=True, help="Zero-based row index to run")
args = parser.parse_args()

csv_path = pathlib.Path(args.csv).resolve()
df = pd.read_csv(csv_path)
row = df.iloc[args.row]
# the code to run this script is:
# python simulation.py --csv oddratio/src_pos_tidy_v2.csv --row 0

# real image pos and dt according to the row specified
real_params = {
    "x_src": float(row["x"]),
    "y_src": float(row["y"]),
}
if 'z' in row.index and pd.notna(row['z']):
    real_params["z_s"] = float(row["z"])
if 'H0' in row.index and pd.notna(row['H0']):
    real_params["H0"] = float(row["H0"])

real_cluster = int(row["indices"])

base_output_dir = pathlib.Path(os.environ.get("OUT_DIR", ".")).resolve()
OUT_DIR = base_output_dir / f"test_tidy_row{args.row}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- standard output columns we'll maintain in-place in the input CSV ----

OUTPUT_DEFAULTS = {
    "run_status": "STARTED",
    "run_msg": "",
    "localized_index": np.nan,   # allow NA; ints will be float in CSV anyway
    "localized_x": np.nan,
    "localized_y": np.nan,
    "localized_z": np.nan,
    "localized_H0": np.nan,
    "chi_sq": np.nan,
    "accepted_clusters": "",
    "out_dir": "",
    "posterior_file": "",
    "corner_plot": "",
    "trace_plot": "",
}
# mark start + OUT_DIR early (useful for debugging)
update_csv_row(csv_path, args.row, {**OUTPUT_DEFAULTS, "out_dir": str(OUT_DIR)})

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
data_sigma_dt_list = []

for i in scenarios:
    clustername = scenarios[i]
    full_cluster_name = full_cluster_names[clustername]

    file_dir = os.getcwd()
    fits_filex = os.path.join(
        file_dir,
        f'GCdata/{full_cluster_name}/cats copy/hlsp_frontier_model_{clustername}_cats_v4_x-arcsec-deflect.fits'
    )
    fits_filey = os.path.join(
        file_dir,
        f'GCdata/{full_cluster_name}/cats copy/hlsp_frontier_model_{clustername}_cats_v4_y-arcsec-deflect.fits'
    )
    psi_file = os.path.join(
        file_dir,
        f'GCdata/{full_cluster_name}/cats copy/hlsp_frontier_model_{clustername}_cats_v4_psi.fits'
    )

    sigma_dt_file = os.path.join(
        file_dir,
        f'GCdata/{full_cluster_name}/cats copy/hlsp_frontier_model_{clustername}_cats_v4_sigma_dt.fits'
    )

    with fits.open(fits_filex) as hdulx, fits.open(fits_filey) as hduly, fits.open(psi_file) as hdul_psi, fits.open(sigma_dt_file) as hdul_sigma_dt:
        datax = hdulx[0].data.astype(np.float32, copy=False)
        datay = hduly[0].data.astype(np.float32, copy=False)
        data_psi = hdul_psi[0].data.astype(np.float32, copy=False)
        data_sigma_dt = hdul_sigma_dt[0].data.astype(np.float32, copy=False)
        
        # Append the data arrays to the lists
        datax_list.append(datax)
        datay_list.append(datay)
        data_psi_list.append(data_psi)
        data_sigma_dt_list.append(data_sigma_dt)

# initialize the LensingData object
pixscale_list = [0.2, 0.3, 0.2, 0.3, 0.8, 0.5]
lensing_data = LensingData(
    alpha_maps_x=datax_list,
    alpha_maps_y=datay_list,
    lens_potential_maps=data_psi_list,
    uncertainty_dt=data_sigma_dt_list,
    pixscale = pixscale_list,
    z_l_list = [0.375, 0.308, 0.351, 0.397, 0.545, 0.543], # Lens redshifts for the two clusters
    # We can use the default x_center, y_center, and search_window_list
    # or override them if needed.
)

#Initialize the Main Analysis Class
z_s_ref = 1.5  # Reference source redshift
cluster_system = ClusterLensing(data=lensing_data, z_s_ref=z_s_ref)

print("Setup complete. Lensing system initialized.")

# Calculate the image positions and time delays for the test parameters
output = cluster_system.calculate_images_and_delays(
    real_params, real_cluster
)

print(f"True time delays: {output['time_delays']}")
dt_true = output['time_delays']

# luminosity distance calculation

if 'H0' not in real_params:
    cosmos = FlatLambdaCDM(H0=70, Om0=0.3)
else:
    cosmos = FlatLambdaCDM(H0=real_params["H0"], Om0=0.3)
lum_dist_true = cosmos.luminosity_distance(real_params['z_s']).value  # Luminosity distance in Mpc
print("True luminosity distances:", lum_dist_true)

# test de then mcmc 
print("\nRunning the full analysis pipeline...")
# Define settings for the MCMC sampler
mcmc_settings = {
    "n_walkers": 20,      # Number of MCMC walkers
    "n_steps": 20000,      # Number of steps per walker
    "fit_z": True,        # We want to fit for the source redshift (z_s)
    "fit_hubble": False,   # We want to fit for the Hubble constant (H0)
    "lum_dist_true": lum_dist_true, # An external "true" luminosity distance measurement (in Mpc)
    "sigma_lum": 0.1,     # The fractional error on the luminosity distance
    # Define the prior boundaries for the parameters being fit in the MCMC
    "z_bounds": (1.0, 5.0),
    "H0_bounds": (60, 80)
}

# Call the main analysis function
# The function will first run DE to find the best cluster,
# then run MCMC on that best-fit model.
mcmc_results, accepted_cluster_indices = cluster_system.find_best_fit(
    dt_true=dt_true,
    run_mcmc=True,
    mcmc_settings=mcmc_settings
)
print(f"\nAnalysis complete. Found {len(mcmc_results)} cluster(s) meeting the MCMC criterion.")
print(f"Accepted cluster indices: {accepted_cluster_indices}")

# collect rows (one per cluster) and write a combined CSV for this source row
combined_rows = []

for result in mcmc_results:
    cluster_idx = result['cluster_index']
    print(f"\n--- Processing Final Results for Cluster {cluster_idx} ---")
    
    # --- Step 5: Analyze and Display ---
    sampler = result['mcmc_sampler']
    burn_in_steps = 3000
    labels = list(result['de_params'].keys())
    flat_samples = sampler.get_chain(discard=burn_in_steps, flat=True)
    
    print(f"--- MCMC Parameter Constraints for Cluster {cluster_idx} ---")
    medians = []
    for i in range(len(labels)):
        mcmc = np.percentile(flat_samples[:, i], [9, 50, 95]) # get the 90% interval by flat_samples
        q = np.diff(mcmc)
        print(f"{labels[i]:>7s} = {mcmc[1]:.3f} +{q[1]:.3f} / -{q[0]:.3f}")
        medians.append(float(np.percentile(flat_samples[:, i], 50)))

    # Save the Localization Results 
    print(f"Saving localization results for Cluster {cluster_idx}...")
    file_path = os.path.join(OUT_DIR, f"cluster_{cluster_idx}_posterior.npz")
    
    cluster_system.save_mcmc_results(
        sampler=sampler,    
        best_result=result, # Pass the individual result dictionary
        n_burn_in=burn_in_steps,
        output_path=file_path,
        dt_true=dt_true,
        mcmc_settings=mcmc_settings
    )

# after running find_best_fit(...)
accepted_str = ",".join(str(i) for i in accepted_cluster_indices)
base_updates = {
    "accepted_clusters": accepted_str,
    "chi_sq": float(min(r['chi_sq'] for r in mcmc_results)) if mcmc_results else 0.0,
    "out_dir": str(OUT_DIR),
}

try:
    write_zero = len(mcmc_results) != 1
    if not write_zero:
        # one cluster passed → take its medians
        result = mcmc_results[0]
        cluster_idx = result['cluster_index']
        sampler = result['mcmc_sampler']
        burn_in_steps = 3000
        labels = list(result['de_params'].keys())
        flat_samples = sampler.get_chain(discard=burn_in_steps, flat=True)
        medians = [float(np.percentile(flat_samples[:, i], 50)) for i in range(len(labels))]

        updates = {
            **base_updates,
            "run_status": "OK",
            "run_msg": "",
            "localized_index": cluster_idx,
            "localized_x": medians[0],
            "localized_y": medians[1],
            "localized_z": medians[2] if len(medians) > 2 else 0.0,
            "localized_H0": medians[3] if len(medians) > 3 else 0.0,
        }
        update_csv_row(csv_path, args.row, updates)
        print(f"Updated CSV row {args.row} with result for cluster {cluster_idx}.")
    else:
        updates = {
            **base_updates,
            "run_status": "NO_FIT",
            "run_msg": f"len(mcmc_results)={len(mcmc_results)}",
            "localized_index": 0,
            "localized_x": 0.0,
            "localized_y": 0.0,
            "localized_z": 0.0,
            "localized_H0": 0.0,
        }
        update_csv_row(csv_path, args.row, updates)
        print(f"Updated CSV row {args.row} with zeros.")
except Exception as e:
    # Defensive: record errors to the CSV as well
    update_csv_row(csv_path, args.row, {
        **base_updates,
        "run_status": "ERROR",
        "run_msg": f"{type(e).__name__}: {e}",
    })
    raise

should_plot = (
    len(accepted_cluster_indices) == 1
    and accepted_cluster_indices[0] == real_cluster
)

if should_plot:
    CLUSTER_INDEX_TO_PLOT = real_cluster
    data_file = os.path.join(OUT_DIR, f"cluster_{CLUSTER_INDEX_TO_PLOT}_posterior.npz")
    if not os.path.exists(data_file):
        print(f"Plot skipped: posterior file not found for cluster {CLUSTER_INDEX_TO_PLOT} at {data_file}")
    else:
        print(f"Loading data from {data_file}...")
        mcmc_data = np.load(data_file)
        flat_chain = mcmc_data['flat_chain']
        full_chain = mcmc_data['chain']
        param_labels = mcmc_data['param_labels']
        truth_values = real_params['x_src'], real_params['y_src'], real_params['z_s'], real_params['H0']

        print("\nGenerating corner plot...")
        fig_corner = corner.corner(
            flat_chain,
            labels=param_labels,
            quantiles=[0.05, 0.5, 0.95],
            show_titles=True,
            truths=truth_values,
            label_kwargs={"fontsize": 14},
            title_kwargs={"fontsize": 12},
            verbose=False
        )
        fig_corner.suptitle(f"Corner Plot for Cluster {CLUSTER_INDEX_TO_PLOT}", fontsize=16)
        corner_plot_path = os.path.join(OUT_DIR, f"cluster_{CLUSTER_INDEX_TO_PLOT}_corner_from_saved.png")
        fig_corner.savefig(corner_plot_path, dpi=150)
        plt.close(fig_corner)
        print(f"-> Corner plot saved to {corner_plot_path}")

        print("\nGenerating trace plot...")
        n_steps, n_walkers, n_dim = full_chain.shape
        fig_trace, axes = plt.subplots(n_dim, figsize=(12, 2 * n_dim), sharex=True)
        steps = np.arange(n_steps)
        for i in range(n_dim):
            ax = axes[i]
            ax.plot(steps, full_chain[:, :, i], alpha=0.2)
            ax.set_ylabel(param_labels[i], fontsize=14)
            ax.tick_params(axis='both', labelsize=12)
        axes[-1].set_xlabel("Step Number", fontsize=14)
        axes[-1].tick_params(axis='both', labelsize=12)
        fig_trace.suptitle("Trace Plot for MCMC parameters", fontsize=16, y=0.99)
        fig_trace.tight_layout(rect=[0, 0, 1, 0.98])
        trace_plot_path = os.path.join(OUT_DIR, f"cluster_{CLUSTER_INDEX_TO_PLOT}_trace.png")
        fig_trace.savefig(trace_plot_path, dpi=150)
        plt.close(fig_trace)
        print(f"-> Trace plot saved to {trace_plot_path}")
        
        update_csv_row(csv_path, args.row, {
            "corner_plot": corner_plot_path,
            "trace_plot": trace_plot_path,
            "posterior_file": data_file,
        })
else:
    print("Plotting skipped: require len(mcmc_results)==1 and accepted_cluster_indices==real_cluster.")