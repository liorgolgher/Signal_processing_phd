# conda activate pysight



#import cv2
import csv
# from numba import jit, njit
import numpy as np
import zarr
from numcodecs import Blosc, Zstd
import time
import dask.array as da
import dask
import glob
import multiprocessing
from tifffile import imread, imsave
import h5py
import pandas as pd
import os

from skimage.transform import downscale_local_mean, rescale


from skimage.transform import radon, iradon


import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.stats import variation


def metroloj_csv_reader(file_name):
    df = pd.read_csv(file_name)

    centerline_depth = np.array(df['Depth']) - 1 # FiJI indices run from 1 rather than 0
    psf_x = np.array(df['X'])
    psf_y = np.array(df['Y'])
    psf_z = np.array(df['Z'])
    # coefficients of determination:
    r2x = np.array(df['R2X']) 
    r2y = np.array(df['R2Y']) 
    r2z = np.array(df['R2Z']) 
    
    return centerline_depth, psf_x, psf_y, psf_z, r2x, r2y, r2z

def is_invalid_psf_fitting(r2_vector, r2_threshold=0.8):
    return r2_vector < r2_threshold

def calc_lateral_psf(raw_psf_x, raw_psf_y, r2x, r2y):
    psf_x = raw_psf_x.copy()
    psf_x[is_invalid_psf_fitting(r2x)] = np.nan
    psf_y = raw_psf_y.copy()
    psf_y[is_invalid_psf_fitting(r2y)] = np.nan
    lateral_psf = np.zeros_like(psf_x)
    for pook in range(len(lateral_psf)):
        try:
            lateral_psf[pook] = np.nanmin(np.array([psf_x[pook],psf_y[pook] ]) )
        except:
            print(f'failure! point_number is {pook} and the psfs are {psf_x[pook]} and {psf_y[pook]}')
    return lateral_psf
    


if __name__ == "__main__":

    data_folder_name = os.path.join('D:','\Downloads')
    data_folder_name = os.path.join(data_folder_name, '2021_02_01_depth_labelled_psfs')

    csv_fn = os.path.join(data_folder_name, '2021_02_01_depth_labelled_psf.csv')

    centerline_depth, raw_psf_x, raw_psf_y, raw_psf_z, r2x, r2y, r2z = metroloj_csv_reader(csv_fn)
    lateral_psf = calc_lateral_psf(raw_psf_x, raw_psf_y, r2x, r2y)
    print(f'lateral psf is {lateral_psf}')
    invalid_axial_psf_vector = is_invalid_psf_fitting(r2z)
    invalid_axial_psf_vector[0] = True
    valid_axial_psf_vector = np.logical_not(invalid_axial_psf_vector)
    invalid_elements = np.logical_or(np.isnan(lateral_psf), invalid_axial_psf_vector)
    print(f'invalid elements are {centerline_depth[invalid_elements]}')

    print(f'First centerline depth is {centerline_depth[0]}, hence its psf invalidity is {invalid_axial_psf_vector[0]}')

    color_by_r2 = np.stack((r2x, r2y, np.ones_like(r2x)))
    num_observations = len(centerline_depth)
    markeredgecolor_array = np.zeros((num_observations,3))
    markeredgecolor_array[invalid_elements,0] = 1
    markeredgecolor_array[np.logical_not(invalid_elements),1] = 1
    print(markeredgecolor_array)

    fig,ax = plt.subplots(1,3)
    for nookvar in range(num_observations):
        ax[0].plot(lateral_psf[nookvar], raw_psf_z[nookvar], '.', markersize=15, markeredgecolor=markeredgecolor_array[nookvar,:])
    ax[0].set_xlabel('Lateral PSF [μm]')
    ax[0].set_ylabel('Axial PSF [μm]')    
    ax[0].set_title('Intravital axial vs. lateral PSF [μm]')    
    
    for nookvar in range(num_observations):
        ax[1].plot(centerline_depth[nookvar], lateral_psf[nookvar],'.', markersize=10, color=color_by_r2[:,nookvar], markeredgecolor=markeredgecolor_array[nookvar,:])
        ax[1].plot(centerline_depth[nookvar], raw_psf_z[nookvar], '.', markersize=20, color=str(r2z[nookvar]), markeredgecolor=markeredgecolor_array[nookvar,:])
    ax[1].set_xlabel('Intravital imaging depth [μm]')
    ax[1].set_ylabel('Intravital PSF [μm]')
    ax[1].set_title('Intravital PSF size [μm] vs. imaging depth [μm]')
    ax[1].legend(['Lateral PSF', 'Axial PSF'])

    ax[2].plot(centerline_depth, r2x, 'o')
    ax[2].plot(centerline_depth, r2y, 'o')
    ax[2].plot(centerline_depth, r2z, 'o')
    plt.hlines(y=[0.8], xmin=[0], xmax=[np.max(centerline_depth)], colors='purple', linestyles='--', lw=2, label='Inclusion threshold')
    ax[2].set_xlabel('Intravital imaging depth [μm]')
    ax[2].set_ylabel('R\u00b2')
    ax[2].legend(['R\u00b2(x)', 'R\u00b2(y)','R\u00b2(z)', 'Inclusion threshold'])
    ax[2].set_title('Coefficients of determination vs. imaging depth [μm]')
    
    plt.show()

