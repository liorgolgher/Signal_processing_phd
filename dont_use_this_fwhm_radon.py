# conda activate cv
import porespy as ps
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt, peak_prominences

import math

from skimage.transform import radon, iradon, iradon_sart
import time
import tifffile as tf

def fwhm_calc(radon_profile):
    # Find the full width at half maximum along a profile in radon space
    radon_profile = radon_profile.astype('float') # convert to float precision to improve division
    radon_profile = radon_profile - np.min(radon_profile)
    
    fwhm_threshold = 0.5 # look for values exceeding 50% of maximal brightness
    over_threshold_indices = np.argwhere(radon_profile > fwhm_threshold*np.max(radon_profile))
    radon_fwhm = over_threshold_indices[-1] - over_threshold_indices[0]
    if(radon_fwhm < 1):
        print('non-positive fwhm values found!')

    return radon_fwhm

def thresholded_radon_profiler(radon_profile):
    # return the thresholded profile in radon space for a specific angle
    radon_profile = radon_profile.astype('float') # convert to float precision to improve division
    radon_profile = radon_profile - np.min(radon_profile)
    fwhm_threshold = 0.5 # look for values exceeding 50% of maximal brightness
    return (radon_profile > fwhm_threshold*np.max(radon_profile))
    
#def calc_vascular_diameter(radon_profile):
def calc_vascular_diameter(radon_profiles):    
    # Find the vascular diameter in real space corresponding to a thresholded radon_profile in Radon space
    '''
    straight_up_radon_transform = np.zeros((radon_profile.shape[0],180))
    straight_up_radon_transform[:,0] = radon_profile
    '''
    straight_up_radon_transform = np.zeros((radon_profiles.shape[0],180))
    a = straight_up_radon_transform[:,89:91]
    #print(a.shape)
    #print(radon_profiles.shape)
    straight_up_radon_transform[:,89:91] = radon_profiles
    straight_up_vessel = iradon_sart(straight_up_radon_transform)
    straight_up_vessel_quartile = np.divide(straight_up_vessel.shape[1], 4).astype('int64')
    straight_vascular_profile = np.sum(straight_up_vessel[:,straight_up_vessel_quartile:-straight_up_vessel_quartile],axis=1)

    

    my_peaks, dict_peak = find_peaks(straight_vascular_profile, prominence=0.0001)
    sort_my_dict = np.argsort(dict_peak["prominences"])
    print(dict_peak["prominences"][sort_my_dict])
    print(my_peaks[sort_my_dict])
    
    #prominences = peak_prominences(straight_vascular_profile, my_peaks)
    #print(my_peaks)

    vascular_diameter = np.abs(my_peaks[sort_my_dict[-1]]-my_peaks[sort_my_dict[-2]])

    return vascular_diameter, straight_up_vessel


def find_vascular_diameter_and_angle(radon_transformed_image):
    # Find the angle for which the FWHM in radon space is minimal, corresponding to minimal vascular section = vascular diameter in real space
    # The image shape assumed to be (n,180)
    radon_fwhm = np.zeros((180,1),dtype='int64')
    for radon_angle in np.arange(180):
        radon_fwhm[radon_angle] = fwhm_calc(radon_transformed_image[:,radon_angle])
    #return np.max(radon_fwhm), np.argmax(radon_fwhm)
    return np.min(radon_fwhm), np.argmin(radon_fwhm)

#def constrained_fwhm_calc(radon_profile, global_threshold, nominal_profile):
def constrained_fwhm_calc(radon_profile, global_threshold=0.5):
# Find the full width at half maximum along the vascular profile in radon space, using the time-collapsed values to constrain the time-varying values
# First use a time-collapsed intensity threshold, then penalize diameters that considerably diverge from the time-collapsed diameter
    over_threshold_indices = np.argwhere(radon_profile > global_threshold)
    constrained_radon_fwhm = over_threshold_indices[-1] - over_threshold_indices[0]
    if(constrained_radon_fwhm < 1):
        print('non-positive fwhm values found!')

    return constrained_radon_fwhm


'''
for t in np.arange(7):
    a = ps.generators.cylinders((200,200,20), radius=10, ncylinders=1, phi_max=0, length=100)
    a = 1-a
    binomial_cylinder = np.random.binomial(1,0.8,size=a.shape)

    peppered_cylinder = np.sum(a*(0.5+binomial_cylinder), axis=2)
    
    
    #raw_cylinder = np.sum(a,axis=2)
    #noisy_cylinder = raw_cylinder * (0.3 + np.random.random(size=(raw_cylinder.shape)))


'''
for filenum in np.arange(6,12):
    filename = 'D:/Downloads/v/' + str(1+filenum) + '.tif'
    peppered_cylinder = tf.imread(filename) 
    
    
    peppered_cylinder = np.pad(peppered_cylinder, np.min(peppered_cylinder.shape))
    radon_tf_cylinder = radon(peppered_cylinder, theta=None, circle=False, preserve_range=False)

    radon_fwhm, radon_angle = find_vascular_diameter_and_angle(radon_tf_cylinder)

    thresholded_radon_transform = np.zeros_like(radon_tf_cylinder)
    #orthogonal_radon_angle = np.remainder(radon_angle+90, 180)

    print(radon_tf_cylinder.shape)
    shifted_radon_transform = np.roll(radon_tf_cylinder,-radon_angle,axis=1)
    '''
    for theta in np.arange(180):
        thresholded_radon_transform[:,theta] = thresholded_radon_profiler(shifted_radon_transform[:,theta])
    #thresholded_radon_transform[:,radon_angle] = thresholded_radon_profiler(radon_tf_cylinder[:,radon_angle])
    '''
    thresholded_radon_transform[:,0] = thresholded_radon_profiler(shifted_radon_transform[:,0])

    deduced_vascular_profile = iradon_sart(thresholded_radon_transform)
    #actual_vascular_diameter, straight_up_vessel = calc_vascular_diameter(thresholded_radon_profiler(radon_tf_cylinder[:,radon_angle]))
    actual_vascular_diameter, straight_up_vessel = calc_vascular_diameter(thresholded_radon_profiler(radon_tf_cylinder[:,89:91]))
    

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)

    ax0.imshow(peppered_cylinder)
    ax1.imshow(radon_tf_cylinder)
    ax1.axvline(x=radon_angle, linestyle=':', color='red', alpha=0.5)
    ax1.set_title('radon fwhm is ' + str(radon_fwhm) + ' pixels along ' + str(radon_angle) + ' degree angle')
    ax2.imshow(deduced_vascular_profile)
    ax3.imshow(straight_up_vessel)
    ax3.set_title('real-space fwhm is ' + str(actual_vascular_diameter) + ' pixels')

    plt.show()

