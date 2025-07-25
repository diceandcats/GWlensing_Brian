# pylint: skip-file
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from lensing_data_class import LensingData
from cluster_local_tidy import ClusterLensing
from cluster_local_new import ClusterLensing_fyp
import pandas as pd
import corner
import arviz as az

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

    # getting the pixel scale list
    pixscale_list = [0.42, 0.51, 0.51, 0.42, 0.42, 0.42]
    lensing_data = LensingData(
    alpha_maps_x=datax_list,
    alpha_maps_y=datay_list,
    lens_potential_maps=data_psi_list,
    pixscale = pixscale_list,
    z_l_list = [0.375, 0.308, 0.348, 0.396, 0.548, 0.544], # Lens redshifts for the two clusters
    # We can use the default x_center, y_center, and search_window_list
    # or override them if needed.
)
    z_s_ref = 1.5 
    cluster_system = ClusterLensing(data=lensing_data, z_s_ref=z_s_ref)

    cluster = ClusterLensing_fyp(datax_list, datay_list, data_psi_list, 0.5, 1, pixscale_list, diff_z=False)

    # de + mcmc with unknown cluster
    parameters = [131,131,3.5,72,2] # x, y, z, H0, index
    dt_obs = cluster.image_and_delay_for_xyzH(parameters[0], parameters[1], parameters[2], parameters[3],parameters[4])[2]
    print("True time delays:", dt_obs)

    test_params = {"x_src" : 131, "y_src": 131, "z_s": 3.50, "H0": 72}
    test_cluster = 2
    output = cluster_system.calculate_images_and_delays(
        test_params, test_cluster
    )
    print(output['time_delays'])
    
    # cosmos = FlatLambdaCDM(H0=parameters[3], Om0=0.3)

    # lum_dist_true = cosmos.luminosity_distance(parameters[2]).value
    # #print(cluster.chi_squared_with_z_Hubble(tuple([105.25810261,85.54573261,3.18181263,72.93719321]), dt_obs, parameters[4], lum_dist_true=lum_dist_true))
    # print("True luminosity distances:", lum_dist_true)

    # opt_pos = None
    # opt_chi_sq = None
    # opt_sampler = None
    # opt_flat_samples = None
    # opt_index = None
    # opt_acceptance_fraction = None

    # n_steps = 8000
    # n_burn_in = 4000

    # try:
    #     for i in range(4,5):
    #         index = i
    #         _, medians, sampler, flat_samples = cluster.localize_diffevo_then_mcmc_known_cluster_Hubble(dt_obs, index,
    #                                         early_stop=0.02,
    #                                         n_walkers=20, n_steps=n_steps, burn_in=n_burn_in,
    #                                         x_range_prior=10.0, y_range_prior=10.0,
    #                                         x_range_int=3.0, y_range_int=3.0, z_range_int=0.5,
    #                                         z_lower=1.0, z_upper=5.0,
    #                                         # Hubble settings
    #                                         H0_lower=43, H0_upper=102,
    #                                         sigma=0.05, sigma_lum=0.05,
    #                                         lum_dist_true=lum_dist_true)

    #         if medians is None:
    #             print("Nothing found for this index.")
    #             continue

    #         flat_samples = sampler.get_chain(discard=n_burn_in, flat=True)
    #         log_probs = sampler.get_log_prob(discard=n_burn_in, flat=True)

    #         # Find best by likelihood
    #         best_idx_ll = np.argmax(log_probs)
    #         best_params_ll = flat_samples[best_idx_ll]

    #         # Choose best_params (here by smallest likelihood)
    #         best_params = best_params_ll
    #         print("Best parameters (by likelihood):", best_params)
    #         chi_sq = cluster.chi_squared_with_z_Hubble(best_params, dt_obs, index, lum_dist_true=lum_dist_true)

    #         # Get the acceptance fraction of the sampler
    #         acceptance_fraction = np.mean(sampler.acceptance_fraction)
    #         if opt_chi_sq is None or chi_sq <= opt_chi_sq:
    #             opt_pos = medians                   # using median for stable estimation
    #             opt_chi_sq = chi_sq
    #             opt_sampler = sampler
    #             opt_flat_samples = flat_samples
    #             opt_index = index
    #             opt_acceptance_fraction = acceptance_fraction
    #             print("Replaced original opt.")

    #             samples_analysis = sampler.get_chain()
    #             log_prob_analysis = sampler.get_log_prob()
    #             idata = az.from_emcee(
    #                 sampler,
    #                 var_names=["x_src", "y_src", "z_s", "H0"],
    #             )
    #             summary = az.summary(
    #                 idata,
    #                 var_names=["x_src", "y_src", "z_s", "H0"],
    #                 round_to=2,
    #             )
    #             print(summary[["mean", "ess_bulk", "r_hat"]])
                
    # except KeyboardInterrupt:
    #     print("Interrupted.")
    #     print("Best fit parameters (median):", opt_pos)
    #     print("Best fit index:", opt_index)
    #     print("Optimized Chi squared value:", opt_chi_sq)
    #     print("samples shape:", opt_flat_samples.shape)
    #     print("Acceptance fraction:", opt_acceptance_fraction)

    #     burn_in = n_burn_in
    #     chain = opt_sampler.get_chain(flat=False)
    #     log_probs = opt_sampler.get_log_prob(discard=burn_in, flat=True)
    #     n_steps, n_walkers, ndim = chain.shape
    #     labels = ["x_src", "y_src", "z_s", "H0"]

    #     # --- Plot Trace Plots ---
    #     fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    #     for i in range(ndim):
    #         ax = axes[i]
    #         for walker in range(n_walkers):
    #             ax.plot(chain[:, walker, i], alpha=0.3)
    #         ax.set_ylabel(labels[i], fontsize = 14)
    #         ax.tick_params(axis='both', labelsize=12)

    #     axes[-1].set_xlabel("Step Number", fontsize = 14)
    #     axes[-1].tick_params(axis='both', labelsize=12)
    #     plt.suptitle("Trace Plots for MCMC Parameters", fontsize=16)
    #     plt.tight_layout()
    #     #plt.savefig('de_mcmc/de_mcmc_trace_fixz.pdf')
    #     plt.show()

    #     # --- Plot Corner Plot ---
    #     # Flatten the chain (each walker’s chain concatenated) after burn-in.
    #     flat_samples = opt_sampler.get_chain(flat=True)

    #     figure = corner.corner(
    #         flat_samples,
    #         labels=labels,
    #         quantiles=[0.05, 0.5, 0.95],  # 90% interval
    #         show_titles=True,
    #         truths=[parameters[0], parameters[1], parameters[2], parameters[3]],  # True values
    #         smooth=1.0,  # Smooth out contours
    #         bins=30,     # Increase the number of bins
    #         label_kwargs={"fontsize": 14},  # Set axis label font size
    #         title_kwargs={"fontsize": 12},  # Set title font size
    #     )

    #     #plt.savefig('de_mcmc/0.05dt/de_mcmc_corner_1.pdf')
    #     plt.show()
    #     exit()


    # print("Best fit parameters (median):", opt_pos)
    # print("Best fit index:", opt_index)
    # print("Optimized Chi squared value:", opt_chi_sq)
    # print("samples shape:", opt_flat_samples.shape)
    # print("Acceptance fraction:", opt_acceptance_fraction)

    # src = pd.read_csv('/home/dices/Research/GWlensing_Brian/src_pos_for_dist_with_z_H0_de+mcmc.csv')
    # # Read the number of rows in the CSV file
    # new_line = len(src)
    # src.at[new_line, 'indices'] = parameters[4]
    # src.at[new_line, 'x'] = parameters[0]
    # src.at[new_line, 'y'] = parameters[1]
    # src.at[new_line, 'z'] = parameters[2]
    # src.at[new_line, 'H0'] = parameters[3]
    # src.at[new_line, 'localized_index'] = opt_index
    # src.at[new_line, 'localized_x'] = opt_pos[0]
    # src.at[new_line, 'localized_y'] = opt_pos[1]
    # src.at[new_line, 'localized_z'] = opt_pos[2]
    # src.at[new_line, 'localized_H0'] = opt_pos[3]
    # src.at[new_line, 'localized_chi_sq'] = opt_chi_sq
    # src.to_csv('/home/dices/Research/GWlensing_Brian/src_pos_for_dist_with_z_H0_de+mcmc.csv', index=False)

    # # Assuming sampler is your emcee sampler object and burn_in is defined.
    # # Retrieve the chain; shape: (n_steps, n_walkers, ndim)
    # burn_in = n_burn_in
    # chain = opt_sampler.get_chain(flat=False)
    # log_probs = opt_sampler.get_log_prob(discard=burn_in, flat=True)
    # n_steps, n_walkers, ndim = chain.shape
    # labels = ["x_src", "y_src", "z_s", "H0"]

    # # --- Plot Trace Plots ---
    # fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    # for i in range(ndim):
    #     ax = axes[i]
    #     for walker in range(n_walkers):
    #         ax.plot(chain[:, walker, i], alpha=0.3)
    #     ax.set_ylabel(labels[i], fontsize = 14)
    #     ax.tick_params(axis='both', labelsize=12)

    # axes[-1].set_xlabel("Step Number", fontsize = 14)
    # axes[-1].tick_params(axis='both', labelsize=12)
    # plt.suptitle("Trace Plots for MCMC Parameters", fontsize=16)
    # plt.tight_layout()
    # #plt.savefig('de_mcmc/de_mcmc_trace_fixz.pdf')
    # plt.show()

    # # --- Plot Corner Plot ---
    # # Flatten the chain (each walker’s chain concatenated) after burn-in.
    # flat_samples = opt_sampler.get_chain(flat=True)

    # figure = corner.corner(
    #     flat_samples,
    #     labels=labels,
    #     quantiles=[0.05, 0.5, 0.95],  # 90% interval
    #     show_titles=True,
    #     truths=[parameters[0], parameters[1], parameters[2], parameters[3]],  # True values
    #     smooth=1.0,  # Smooth out contours
    #     bins=30,     # Increase the number of bins
    #     label_kwargs={"fontsize": 14},  # Set axis label font size
    #     title_kwargs={"fontsize": 12},  # Set title font size
    # )

    # #plt.savefig('de_mcmc/de_mcmc_corner_fixz.pdf')
    # plt.show()
