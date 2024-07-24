# Milk temporal dynamics from dummy vessels in previously segemented volumes of interest, preparing them for downstream analysis
# conda activate pysight


import csv
from numba import jit, njit
import numpy as np
from numpy.random import default_rng, binomial, randint

import zarr
from numcodecs import Blosc, Zstd
import time
import dask.array as da
import dask
import glob
import multiprocessing
from tifffile import imsave
import h5py
import pandas as pd


from skimage.transform import downscale_local_mean, rescale, rotate


from skimage.transform import radon, iradon


import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, find_peaks, sosfilt
from scipy.stats import variation
import evaluate_radon_dynamics_milker

def generate_diameter_dynamics(num_frames = 1000, sampling_frequency_hz=float(30)):
 num_added_frames = 300 # add frames to the beginning of the vector, accounting for transients

 slow_magnitude = 13
#generate a 1D vector of diameter changes over time
 sos = butter(4,0.15, 'low', output='sos', fs=sampling_frequency_hz)

 #low pass filter, with cutoff at f =2hz
 sos_broader = butter(4,1, 'low', output='sos', fs=sampling_frequency_hz)

 raw_input_diameter = 1 + np.abs(rng.normal(50, 12, size=(num_frames+num_added_frames,)))

 smoothed_trace = slow_magnitude*sosfilt(sos, raw_input_diameter) + sosfilt(sos_broader, raw_input_diameter)

 return smoothed_trace[num_added_frames:]



def generate_dummy_vessel(diameter_dynamics, rotation_angle=0, pixel_density = 0.02, random_noise_factor = 100):
    num_axial_slices = 20
    vessel_segment_length = 170
    white_region_width = 200
    mask_width = 2 * white_region_width
    frame_width = 70
    num_frames = len(diameter_dynamics)

    black_white_mask = np.zeros((vessel_segment_length, mask_width), dtype='float')
    black_white_mask[:,white_region_width:] = 2
    #black_white_mask[:,:white_region_width] = 1
    #black_white_mask = np.fliplr(black_white_mask)
    black_white_mask = np.cumsum(black_white_mask, axis=1)
    black_white_mask = np.sqrt(black_white_mask)
    black_white_mask = np.fliplr(black_white_mask)

    '''
    plt.imshow(black_white_mask)
    plt.title('black and white mask')
    plt.show()
    '''

    radius_dynamics = np.floor(diameter_dynamics/2).astype('int32')



    dummy_vessel = np.zeros((vessel_segment_length, 2*frame_width, num_frames),dtype='float')
    random_sampling_grid = binomial(num_axial_slices,pixel_density,dummy_vessel.shape)

    random_noise = binomial(num_axial_slices,random_noise_factor*pixel_density,dummy_vessel.shape) # simulating the summation of num_axial_slices axial slices, each with its independent binomial sampling

    


    for frame_number in range(num_frames):
        half_vessel = black_white_mask[:,-white_region_width-radius_dynamics[frame_number]:-white_region_width-radius_dynamics[frame_number]+frame_width]
        other_half = np.fliplr(half_vessel)
        dummy_vessel[:,:,frame_number] = np.concatenate((other_half,half_vessel), axis=1)

        '''
        plt.imshow(dummy_vessel[:,:,frame_number])
        plt.title('dummy vessel with width ' + str(np.sum(dummy_vessel[0,:,frame_number])) + ' pixels and expected width ' + str(diameter_dynamics[frame_number]))
        #plt.colorbar()
        plt.show()
        '''
    big_picture = np.floor((rotate(dummy_vessel, rotation_angle) + random_noise) * random_sampling_grid) 
    big_picture_quadrants = np.divide(big_picture.shape, 8).astype('int32')

    return big_picture[big_picture_quadrants[0]:7*big_picture_quadrants[0],big_picture_quadrants[1]:7*big_picture_quadrants[1],:] # crop the medial 60% of the image in x and y

if __name__=='__main__':
    rng = default_rng()

    num_frames = 2700
    frame_rate = 30 # [Hz]
    random_noise_factor = 0.005 # 100
    pixel_density = 0.005 # 0.0005
    random_noise_pixel_density = np.minimum(0.05, random_noise_factor*pixel_density)

    raw_dummy_dynamics = generate_diameter_dynamics(num_frames, frame_rate)
    normalized_dummy_dynamics = raw_dummy_dynamics * 50 * np.reciprocal(np.mean(raw_dummy_dynamics))
    discretized_dummy_dynamics = (normalized_dummy_dynamics/2).astype('int32') * 2
    vessel_rotation_angle = float(randint(180))

    '''
    plt.plot(discretized_dummy_dynamics)
    plt.title(str(discretized_dummy_dynamics.shape))
    plt.show()
    '''

    
    

    dummy_vessel = generate_dummy_vessel(discretized_dummy_dynamics, vessel_rotation_angle, pixel_density, random_noise_factor)


    
    print(f'dummy vessel shape is {dummy_vessel.shape}')
    dummy_vessel_photon_count = np.sum(np.mean(dummy_vessel.astype('float'), axis=0), axis=0) / pixel_density
    dummy_vessel_photon_count_variations = 100 * (dummy_vessel_photon_count / np.mean(dummy_vessel_photon_count) - 1)
    dummy_vessel_simulated_variations = 100 * (discretized_dummy_dynamics / np.mean(discretized_dummy_dynamics) - 1)

    time_vector = np.linspace(0,float(num_frames)/float(frame_rate), num_frames)
    print(dummy_vessel_photon_count.shape)
    '''
    plt.plot(time_vector, dummy_vessel_simulated_variations, alpha=0.5)
    plt.plot(time_vector, dummy_vessel_photon_count_variations, alpha=0.5)
    plt.title('input and output vessel diameter variations [%]')
    plt.xlabel('Time [s]')
    plt.ylabel('diameter changes [%]')
    '''



    # Now that the simulated vascular dynamics are ready, let's try to measure them using functions from evaluate_radon_dynamics_milker.py:
    lowpass_frequency = 1 # [Hz]


    sos_main = butter(4, lowpass_frequency, 'lp', fs=frame_rate, output='sos')
    low_passed_vessel = sosfiltfilt(sos_main, dummy_vessel, axis=2)

    time_collapsed_vessel_lp = np.sum(low_passed_vessel, axis=2)
    radon_transformed_tc_vessel_lp = evaluate_radon_dynamics_milker.radon(time_collapsed_vessel_lp)
    vascular_angle_estimate = evaluate_radon_dynamics_milker.find_vascular_angle(radon_transformed_tc_vessel_lp)

    fig4,axs4, = plt.subplots(1,2)

    axs4[0].imshow(time_collapsed_vessel_lp, cmap='gray')
    axs4[0].set_title(f'time collapsed vessel with angle {vessel_rotation_angle} and estimated angle {vascular_angle_estimate}')


    photon_count_lp = np.sum(low_passed_vessel, axis=(0,1))
    print('shape of low_passed_vessel is: '+ str(low_passed_vessel.shape))

    time_collapsed_radon_profile = radon(time_collapsed_vessel_lp, theta=(vascular_angle_estimate,), circle=False, preserve_range=False)
    '''
    wrong_time_collapsed_radon_profile = radon(time_collapsed_vessel_lp, theta=(vascular_angle_estimate+90,), circle=False, preserve_range=False) # calculating the orthogonal radon profile 
    plt.plot(time_collapsed_radon_profile, alpha=0.8)
    plt.plot(wrong_time_collapsed_radon_profile, alpha=0.2)
    plt.title('time collapsed radon profile - parallel and orthogonal')
    plt.show()
    '''

    global_radon_threshold = evaluate_radon_dynamics_milker.calc_global_radon_threshold(time_collapsed_radon_profile)
    global_radon_threshold = np.divide(global_radon_threshold, num_frames)
    print(f'global radon threshold is {global_radon_threshold}')

    time_varying_radon_diameter = np.zeros((num_frames,))

    for frame_number in range(num_frames):
        momentary_radon_profile = radon(np.squeeze(low_passed_vessel[:,:,frame_number]), theta=(vascular_angle_estimate,), circle=False, preserve_range=False)
        time_varying_radon_diameter[frame_number] = evaluate_radon_dynamics_milker.dynamic_fwhm_calc(momentary_radon_profile, global_radon_threshold)
    

    snapshot_frame_number = 20
    axs4[1].imshow(dummy_vessel[:,:,snapshot_frame_number], cmap='gray')
    #axs4[1].set_title(f'dummy vessel with simulated width {discretized_dummy_dynamics[snapshot_frame_number]} pixels and estimated width {time_varying_radon_diameter[snapshot_frame_number]} pixels at frame number {snapshot_frame_number}')
    axs4[1].set_title(f'Dummy vessel with pixel density of {np.around(pixel_density*100, decimals=2)} % and random noise of {np.around(random_noise_factor*100, decimals=2)} %')



    fig1,axs1, = plt.subplots(1,1)

    '''
    axs1.plot(time_vector, discretized_dummy_dynamics, alpha=0.5, label='Simulated')
    axs1.plot(time_vector, time_varying_radon_diameter, alpha=0.5, label='Radon')
    #plt.plot(time_vector, photon_count_lp, alpha=0.3, label='photon_counting')
    axs1.set_title('simulated and estimated vessel diameter [pixels]')
    axs1.set_xlabel('Time [s]')
    axs1.set_ylabel('vessel diameter [pixels]')
    axs1.legend()
    plt.show()
    '''    

    dummy_vessel_simulated_variations = 100 * (discretized_dummy_dynamics / np.mean(discretized_dummy_dynamics) - 1)
    radon_diameter_variations = 100 * (time_varying_radon_diameter / np.mean(time_varying_radon_diameter) - 1)

    axs1.plot(time_vector, dummy_vessel_simulated_variations, alpha=0.5, label='Simulated')
    axs1.plot(time_vector, radon_diameter_variations, alpha=0.5, label='Estimated')
    #plt.plot(time_vector, photon_count_lp, alpha=0.3, label='photon_counting')
    axs1.set_title('Simulated and estimated vessel diameter variations [%]')
    axs1.set_xlabel('Time [s]')
    axs1.set_ylabel('diameter changes [%]')
    axs1.legend()
    
    plt.show()
    

    



    


