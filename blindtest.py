"""blind test code"""
import os
import numpy as np
from astropy.io import fits
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from astropy.cosmology import FlatLambdaCDM


print("Please input the index(0-5) of the name of the galaxy cluster for placing the source listed: \nAbell 370, Abell 2744, Abell S1063, MACS0416, MACS0717, MACS1149")
cluster_index = input()
print("Please input the source and lens redshifts separated by a comma: ")
z_s, z_l = input().split(',')
z_s = float(z_s)
z_l = float(z_l)

# 6 cases
scenarios = {
    '0': 'abell370',
    '1': 'abell2744',
    '2': 'abells1063',
    '3': 'macs0416',
    '4': 'macs0717',
    '5': 'macs1149'
}

full_cluster_names = {
    'abell370': 'Abell 370',
    'abell2744': 'Abell 2744',
    'abells1063': 'Abell S1063',
    'macs0416': 'MACS J0416.1-2403',
    'macs0717': 'MACS J0717.5+3745',
    'macs1149': 'MACS J1149.5+2223'
}

if cluster_index in scenarios:
    clustername = scenarios[cluster_index]
    full_cluster_name = full_cluster_names[clustername]

    file_dir = os.path.dirname(os.path.realpath(__file__))
    fits_filex = os.path.join(file_dir,f'GCdata/{full_cluster_name}/hlsp_frontier_model_{clustername}_williams_v4_x-arcsec-deflect.fits')
    fits_filey = os.path.join(file_dir,f'GCdata/{full_cluster_name}/hlsp_frontier_model_{clustername}_williams_v4_y-arcsec-deflect.fits')
    psi_file = os.path.join(file_dir,f'GCdata/{full_cluster_name}/hlsp_frontier_model_{clustername}_williams_v4_psi.fits')
    
    hdul = fits.open(fits_filex)
    hdul1 = fits.open(fits_filey)
    hdul_psi = fits.open(psi_file)

    datax = hdul[0].data
    datay = hdul1[0].data
    data_psi = hdul_psi[0].data
    hdul.close()
    hdul1.close()
    hdul_psi.close()

    def get_pixscale(cluster_name, file_path='GCdata/pixsize'):
        full_path = os.path.join(file_dir, file_path)
        with open(full_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.startswith(cluster_name):
                    # Split the line to get the value after the colon and return it as a float
                    return float(line.split(':')[1].strip())
        return None  # Return None if the cluster name isn't found
    pixscale = get_pixscale(full_cluster_name)
    data_psi_arcsec = data_psi * pixscale**2

    realsize = datax.shape[0]
    grid = np.linspace(0, realsize-1, realsize) * pixscale

    # scal the maps with input redshifts
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    D_LS = cosmo.angular_diameter_distance_z1z2(z_l, z_s)
    D_S = cosmo.angular_diameter_distance(z_s)

    # Scale deflection map
    scal = D_LS/D_S
    datax *= scal
    datay *= scal
    data_psi_arcsec *= scal
    print(f'Scaling factor: {scal}')

    lens_model_list = ['INTERPOL']
    kwargs_lens = [{'grid_interp_x': grid, 'grid_interp_y': grid, 'f_': data_psi_arcsec,
                            'f_x': datax, 'f_y': datay}]
    lensModel = LensModel(lens_model_list=lens_model_list, z_lens=z_l,
        z_source=z_s)
    solver = LensEquationSolver(lensModel)

else:
    print("Invalid input")
    exit()

print(f'Please input the source position in the format x,y (range of the values:0 to {(realsize-1)*pixscale})')
src_pos = input()
src_pos = src_pos.split(',')
src_pos = list(map(float, src_pos))
#print(src_pos)

x_center, y_center = [90,75,110,70,90,70], [70,80,95,60,93,65]
img_pos = solver.image_position_from_source(src_pos[0], src_pos[1], kwargs_lens,
                                            min_distance=pixscale, search_window=100,
                                            verbose=False, x_center=x_center[int(cluster_index)], y_center=y_center[int(cluster_index)])
#print(f'Image positions: {img_pos}')
mag = lensModel.magnification(img_pos[0], img_pos[1], kwargs_lens)
#print(f'Magnification: {mag}')
t = lensModel.arrival_time(img_pos[0], img_pos[1], kwargs_lens,
                           x_source=src_pos[0], y_source=src_pos[1])
dt = t -t.min()
#print(f'Time delay: {dt}')

print(f'Infomations for test:\nNumber of images: {len(img_pos[0])}\nMagnification: {mag}\nTime delay: {dt}')
print(f'Image positions: {img_pos}')
# test case: src: 63.8, 84.1 abell 2744