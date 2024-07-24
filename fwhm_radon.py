# conda activate cv
import porespy as ps
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt, peak_prominences, gaussian
from scipy.stats import variation
from scipy.ndimage import convolve1d

import math

from skimage.transform import radon, iradon, iradon_sart, rotate
import time
import tifffile as tf

def fwhm_calc(radon_profile):
    # Find the full width at half maximum along a profile in radon space
    radon_profile = radon_profile.astype('float') # convert to float precision to improve division
    radon_profile = radon_profile - np.nanmin(radon_profile)
    
    fwhm_threshold = 0.5 # look for values exceeding 50% of maximal brightness
    over_threshold_indices = np.argwhere(radon_profile > fwhm_threshold*np.nanmax(radon_profile))
    radon_fwhm = over_threshold_indices[-1] - over_threshold_indices[0]
    if(radon_fwhm < 1):
        print('non-positive FWHM values found!')
        radon_fwhm = np.nan

    return radon_fwhm

def thresholded_radon_profiler(radon_profile):
    # return the thresholded profile in radon space for a specific angle
    radon_profile = radon_profile.astype('float') # convert to float precision to improve division
    radon_profile = radon_profile - np.nanmin(radon_profile)
    fwhm_threshold = 0.5 # look for values exceeding 50% of maximal brightness
    return (radon_profile > fwhm_threshold*np.max(radon_profile))
    
def calc_vascular_diameter(radon_profile):
    # Find the vascular diameter in real space corresponding to a thresholded radon_profile in Radon space
    straight_up_radon_transform = np.zeros((radon_profile.size,180))
    straight_up_radon_transform[:,0] = radon_profile
    straight_up_vessel = iradon(straight_up_radon_transform)
    straight_up_vessel_quartile = np.divide(straight_up_vessel.shape[0], 4).astype('int64')
    straight_vascular_profile = np.sum(straight_up_vessel[straight_up_vessel_quartile:-straight_up_vessel_quartile,:],axis=0)
    diff_straight_vascular_profile = np.abs(np.diff(straight_vascular_profile)) # the greatest difference between neighbouring pixels is expected at the edges of the radon profile
    height_threshold = np.divide(np.max(diff_straight_vascular_profile),100)

    my_peaks, dict_peak = find_peaks(diff_straight_vascular_profile, height=height_threshold) 
    sort_my_dict = np.argsort(dict_peak["peak_heights"])
    #print(dict_peak["prominences"][sort_my_dict])
    print(my_peaks[sort_my_dict])
    print(my_peaks[sort_my_dict[-3:-1]])

    

    vascular_diameter = np.abs(my_peaks[sort_my_dict[-1]]-my_peaks[sort_my_dict[-2]])

    return vascular_diameter, straight_up_vessel

def find_vascular_angle(radon_transformed_image):
    # Find the angle for which the coefficient of variation in Radon space is maximal, corresponding to minimal vascular section = vascular diameter in real space
    # The image shape assumed to be (n,180)
    variation_in_radon_space = variation(radon_transformed_image, axis=0, nan_policy='omit')

    try:
      angle_of_max_cv = np.nanargmax(variation_in_radon_space)
    except:
      angle_of_max_cv = np.nan
      print('vessel orientation could not be determined. Returning nan value.')
    return angle_of_max_cv

def find_vascular_diameter_and_angle(radon_transformed_image):
    # Find the angle for which the FWHM in radon space is minimal, corresponding to minimal vascular section = vascular diameter in real space
    # The image shape assumed to be (n,180)
    radon_fwhm = np.zeros((180,1),dtype='int64')
    for radon_angle in np.arange(180):
        radon_fwhm[radon_angle] = fwhm_calc(radon_transformed_image[:,radon_angle])

    return np.nanmin(radon_fwhm), np.nanargmin(radon_fwhm)


def gaussian_functional(x,a,x0,sigma):
    return np.abs(a)*np.exp(-(x-x0)**2/(2*sigma**2)) # constraining the background_level and the gaussian amplitude to be non-negative

def gaussian_fit(radon_profile, nominal_diameter):
  profile_length = len(radon_profile)
  x_range = np.arange(0,profile_length)
  # setting initial values:
  initial_guess = np.zeros((3,))
  initial_guess[0] = np.max(radon_profile) # amplitude of Gaussian
  initial_guess[1] = np.argmax(radon_profile) # center of Gaussian
  initial_guess[2] = np.divide(nominal_diameter, 2*np.sqrt(2*np.log(2)) ) # relationship between FWHM and variance of Gaussian
  initial_guess = np.squeeze(initial_guess) # initial_guess must have at most 1 dimension

  # setting bounds:
  bounds=((0, 0, initial_guess[2]*0.7), (np.inf, len(radon_profile), initial_guess[2]*1.3))

  x_range = np.arange(0,len(radon_profile))
  try:
    p_optimal, p_cov = curve_fit(gaussian_functional,x_range,radon_profile,p0=initial_guess, bounds=bounds)
  except:
    p_optimal = [np.nan, np.nan, np.nan] # if the optimization fails
  return p_opt, p_cov


#def constrained_fwhm_calc(radon_profile, global_threshold, nominal_profile):
def constrained_fwhm_calc(radon_profile, global_threshold=0.5):
# Find the full width at half maximum along the vascular profile in radon space, using the time-collapsed values to constrain the time-varying values
# First use a time-collapsed intensity threshold, then penalize diameters that considerably diverge from the time-collapsed diameter
    over_threshold_indices = np.argwhere(radon_profile > global_threshold)
    constrained_radon_fwhm = over_threshold_indices[-1] - over_threshold_indices[0]
    if(constrained_radon_fwhm < 1):
        print('non-positive FWHM values found!')

    return constrained_radon_fwhm


#'''
#convolution_kernel = gaussian(300,std=50)
convolution_kernel = gaussian(100,std=100)
plt.plot(convolution_kernel)
plt.show()

for t in np.arange(7):
    a = ps.generators.cylinders((400,400,200), radius=50, ncylinders=1, phi_max=0, length=500)
    a = 1-a
    binomial_cylinder = np.random.binomial(1,0.95,size=a.shape)
    background_noise =  np.random.rand(*a.shape)

    # accounting for extended PSF in z that exceeds the vessel diameter
    xyz_cylinder = convolve1d(0.0003*background_noise + a*(0 + binomial_cylinder), weights=convolution_kernel, axis=2, mode='constant', cval=0.0)
    peppered_cylinder = np.sum(xyz_cylinder,axis=2)
    

    
    
    #raw_cylinder = np.sum(a,axis=2)
    #noisy_cylinder = raw_cylinder * (0.3 + np.random.random(size=(raw_cylinder.shape)))

    '''
for filenum in np.arange(1,11):
    filename = 'D:/Downloads/v/' + str(1+filenum) + '.tif'
    peppered_cylinder = tf.imread(filename) 
        
    #'''
    #peppered_cylinder = np.pad(peppered_cylinder, np.min(peppered_cylinder.shape))
    radon_tf_cylinder = radon(peppered_cylinder, theta=None, circle=False, preserve_range=False)

    #radon_fwhm, radon_angle = find_vascular_diameter_and_angle(radon_tf_cylinder)
    radon_angle = find_vascular_angle(radon_tf_cylinder)

    thresholded_radon_transform = np.zeros_like(radon_tf_cylinder)
    orthogonal_radon_angle = np.remainder(radon_angle+90, 180)
    thresholded_radon_transform[:,radon_angle] = thresholded_radon_profiler(radon_tf_cylinder[:,radon_angle])
    # thresholded_radon_transform[:,orthogonal_radon_angle] = thresholded_radon_profiler(radon_tf_cylinder[:,orthogonal_radon_angle])
    deduced_vascular_profile = iradon(thresholded_radon_transform)
    actual_vascular_diameter, straight_up_vessel = calc_vascular_diameter(thresholded_radon_transform[:,radon_angle])
    
    correction_factor = 1.2 # FWHM diameter is typically 1/1.2 smaller than the actual diameter in ideal cases
    micron_to_pixel_ratio = np.divide(543,2048)
    actual_vascular_diameter = correction_factor * actual_vascular_diameter * micron_to_pixel_ratio
    


    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)

    ax0.imshow(peppered_cylinder)
    ax1.imshow(radon_tf_cylinder)
    ax1.axvline(x=radon_angle, linestyle=':', color='red', alpha=0.5)
    ax1.set_title('Maximal variation at ' + str(radon_angle) + '\N{DEGREE SIGN} angle')
    ax2.imshow(deduced_vascular_profile)
    ax2.set_title('Thresholded profile along Radon angle, transformed to real Space')
    ax3.imshow(straight_up_vessel)
    ax3.set_title(f'Real-space diameter is ' + str(np.round(actual_vascular_diameter, 1)) + '\u03BCm')

    plt.show()

    centerpoint_of_straight_up_vessel = np.divide(straight_up_vessel.shape[0], 2).astype('int64')
    rotated_cylinder = rotate(peppered_cylinder, -radon_angle)
    centerpoint_of_rotated_cylinder = np.divide(rotated_cylinder.shape[0], 2).astype('int64')
    profile_of_straight_up_vessel = np.sum(straight_up_vessel, axis=0)
    profile_of_rotated_cylinder = np.sum(rotated_cylinder, axis=0)

    fig2, ((ax4, ax5), (ax6, ax7)) = plt.subplots(2, 2)
    

    #ax4.plot(radon_tf_cylinder[:,radon_angle])
    ax4.plot(thresholded_radon_transform[:,radon_angle])
    ax4.set_title('Thresholded profile in Radon Space')
    #ax5.plot(straight_up_vessel[centerpoint_of_straight_up_vessel,:])
    ax5.plot(np.abs(np.diff(profile_of_straight_up_vessel)))
    ax5.set_title('Absolute difference of - thresholded profile transformed to real Space')
    ax6.imshow(rotated_cylinder)
    ax6.set_title('rotated vessel in real space')
    #ax7.plot(rotated_cylinder[centerpoint_of_rotated_cylinder,:])
    ax7.plot(profile_of_rotated_cylinder)
    ax7.set_title('profile of rotated vessel in real space')
    plt.show()

'''
    fig3, ((ax8, ax9), (ax10, ax11)) = plt.subplots(2, 2)
    
    ax8.imshow(peppered_cylinder)
    ax9.imshow(np.sum(xyz_cylinder, axis=0))
    ax10.imshow(np.sum(xyz_cylinder, axis=1))
    ax11.imshow(np.squeeze(xyz_cylinder[:,100,:]))
    plt.show()
'''