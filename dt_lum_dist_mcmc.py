# pylint: skip-file
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from math import ceil, floor
from scipy.optimize import minimize
from tqdm import tqdm
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from cluster_local_new import ClusterLensing_fyp
import pandas as pd
import corner

if __name__ == "__main__":
    # inject data

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
            f'Research/GWlensing_Brian/GCdata/{full_cluster_name}/hlsp_frontier_model_{clustername}_williams_v4_x-arcsec-deflect.fits'
        )
        fits_filey = os.path.join(
            file_dir,
            f'Research/GWlensing_Brian/GCdata/{full_cluster_name}/hlsp_frontier_model_{clustername}_williams_v4_y-arcsec-deflect.fits'
        )
        psi_file = os.path.join(
            file_dir,
            f'Research/GWlensing_Brian/GCdata/{full_cluster_name}/hlsp_frontier_model_{clustername}_williams_v4_psi.fits'
        )

        with fits.open(fits_filex) as hdulx, fits.open(fits_filey) as hduly, fits.open(psi_file) as hdul_psi:
            datax = hdulx[0].data
            datay = hduly[0].data
            data_psi = hdul_psi[0].data

            # Append the data arrays to the lists
            datax_list.append(datax)
            datay_list.append(datay)
            data_psi_list.append(data_psi)

    # getting the pixel scale list
    pixscale_list = [0.2, 0.25, 0.25, 0.2, 0.2, 0.2]
    cluster = ClusterLensing_fyp(datax_list, datay_list, data_psi_list, 0.5, 1, pixscale_list, diff_z=False)

    # de + mcmc with unknown cluster

    parameters = [72.4,57.2,2.9,73,3] # x, y, z, H0, index
    dt_obs = cluster.image_and_delay_for_xyzH(parameters[0], parameters[1], parameters[2], parameters[3],parameters[4])[2]
    print("True time delays:", dt_obs)
    cosmos = FlatLambdaCDM(H0=parameters[3], Om0=0.3)
    lum_dist_true = cosmos.luminosity_distance(parameters[2]).value
    print("True luminosity distances:", lum_dist_true)

    opt_pos = None
    opt_chi_sq = None
    opt_sampler = None
    opt_flat_samples = None
    opt_index = None
    opt_acceptance_fraction = None

    n_steps = 40000
    n_burn_in = 20000

    try:
        for i in range(6):
            index = i
            _, medians, sampler, flat_samples = cluster.localize_diffevo_then_mcmc_known_cluster_Hubble(dt_obs, index,
                                            early_stop=0.02,
                                            n_walkers=20, n_steps=n_steps, burn_in=n_burn_in,
                                            x_range_prior=10.0, y_range_prior=10.0,
                                            x_range_int=3.0, y_range_int=3.0, z_range_int=0.5,
                                            z_lower=1.0, z_upper=5.0,
                                            # Hubble settings
                                            H0_lower=43, H0_upper=102,
                                            sigma=0.05, sigma_lum=0.05,
                                            lum_dist_true=lum_dist_true)

            if medians is None:
                print("Nothing found for this index.")
                continue

            flat_samples = sampler.get_chain(discard=n_burn_in, flat=True)
            log_probs = sampler.get_log_prob(discard=n_burn_in, flat=True)

            # Find best by likelihood
            best_idx_ll = np.argmax(log_probs)
            best_params_ll = flat_samples[best_idx_ll]

            # Choose best_params (here by smallest likelihood)
            best_params = best_params_ll
            print("Best parameters (by likelihood):", best_params)
            chi_sq = cluster.chi_squared_with_z_Hubble(best_params, dt_obs, index, lum_dist_true=lum_dist_true)

            # Get the acceptance fraction of the sampler
            acceptance_fraction = np.mean(sampler.acceptance_fraction)
            if opt_chi_sq is None or chi_sq <= opt_chi_sq:
                opt_pos = best_params
                opt_chi_sq = chi_sq
                opt_sampler = sampler
                opt_flat_samples = flat_samples
                opt_index = index
                opt_acceptance_fraction = acceptance_fraction
                print("Replaced original opt.")
                
    except KeyboardInterrupt:
        print("Interrupted.")
        print("Best fit parameters:", opt_pos)
        print("Best fit index:", opt_index)
        print("Optimized Chi squared value:", opt_chi_sq)
        print("samples shape:", opt_flat_samples.shape)
        print("Acceptance fraction:", opt_acceptance_fraction)

        burn_in = n_burn_in
        chain = opt_sampler.get_chain(flat=False)
        log_probs = opt_sampler.get_log_prob(discard=burn_in, flat=True)
        n_steps, n_walkers, ndim = chain.shape
        labels = ["x_src", "y_src", "z_s", "H0"]

        # --- Plot Trace Plots ---
        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        for i in range(ndim):
            ax = axes[i]
            for walker in range(n_walkers):
                ax.plot(chain[:, walker, i], alpha=0.3)
            ax.set_ylabel(labels[i])
        axes[-1].set_xlabel("Step Number")
        plt.suptitle("Trace Plots for MCMC Parameters", fontsize=16)
        plt.tight_layout()
        #plt.savefig('de_mcmc/0.05dt/de_mcmc_trace_1.pdf')
        plt.show()

        # --- Plot Corner Plot ---
        # Flatten the chain (each walker’s chain concatenated) after burn-in.
        flat_samples = opt_sampler.get_chain(flat=True)

        figure = corner.corner(
            flat_samples,
            labels=labels,
            quantiles=[0.05, 0.5, 0.95],  # 90% interval
            show_titles=True,
            truths=[parameters[0], parameters[1], parameters[2], parameters[3]],  # True values
            smooth=1.0,  # Smooth out contours
            bins=30,     # Increase the number of bins
        )

        #plt.savefig('de_mcmc/0.05dt/de_mcmc_corner_1.pdf')
        plt.show()
        exit()


    print("Best fit parameters:", opt_pos)
    print("Best fit index:", opt_index)
    print("Optimized Chi squared value:", opt_chi_sq)
    print("samples shape:", opt_flat_samples.shape)
    print("Acceptance fraction:", opt_acceptance_fraction)

    src = pd.read_csv('/home/dices/Research/GWlensing_Brian/src_pos_for_dist_with_z_de+mcmc.csv')
    src.at[i, 'indices'] = parameters[3]
    src.at[i, 'x'] = parameters[0]
    src.at[i, 'y'] = parameters[1]
    src.at[i, 'z'] = parameters[2]
    src.at[i, 'localized_index'] = opt_index
    src.at[i, 'localized_x'] = opt_pos[0]
    src.at[i, 'localized_y'] = opt_pos[1]
    src.at[i, 'localized_z'] = opt_pos[2]
    src.at[i, 'localized_chi_sq'] = opt_chi_sq
    src.to_csv('/home/dices/Research/GWlensing_Brian/src_pos_for_dist_with_z_de+mcmc.csv', index=False)

    # Assuming sampler is your emcee sampler object and burn_in is defined.
    # Retrieve the chain; shape: (n_steps, n_walkers, ndim)
    burn_in = n_burn_in
    chain = opt_sampler.get_chain(flat=False)
    log_probs = opt_sampler.get_log_prob(discard=burn_in, flat=True)
    n_steps, n_walkers, ndim = chain.shape
    labels = ["x_src", "y_src", "z_s", "H0"]

    # --- Plot Trace Plots ---
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        for walker in range(n_walkers):
            ax.plot(chain[:, walker, i], alpha=0.3)
        ax.set_ylabel(labels[i])
    axes[-1].set_xlabel("Step Number")
    plt.suptitle("Trace Plots for MCMC Parameters", fontsize=16)
    plt.tight_layout()
    #plt.savefig('de_mcmc/de_mcmc_trace_fixz.pdf')
    plt.show()

    # --- Plot Corner Plot ---
    # Flatten the chain (each walker’s chain concatenated) after burn-in.
    flat_samples = opt_sampler.get_chain(flat=True)

    figure = corner.corner(
        flat_samples,
        labels=labels,
        quantiles=[0.05, 0.5, 0.95],  # 90% interval
        show_titles=True,
        truths=[parameters[0], parameters[1], parameters[2], parameters[3]],  # True values
        smooth=1.0,  # Smooth out contours
        bins=30,     # Increase the number of bins
    )

    #plt.savefig('de_mcmc/de_mcmc_corner_fixz.pdf')
    plt.show()
