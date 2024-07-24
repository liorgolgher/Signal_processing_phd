# assign a separate binary mask to each segment
# conda activate vg

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:10:21 2019

@author: rdamseh
"""

import skimage.io as skio
from matplotlib import pyplot as plt
import numpy as np
#import cv2
import VascGraph as vg
from tqdm import tqdm
import h5py
import zarr
import multiprocessing
import os
import tifffile as tf
from skimage.transform import radon, iradon
from numcodecs import Blosc, Zstd
import time
import dask.array as da
import dask
import glob
#from numba import jit, njit
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.stats import variation
from skimage.transform import downscale_local_mean, rescale


def save_dataset(save_file_name,group_name,dataset):
  root = zarr.open_group(save_file_name, mode='a')
  fill_me = root.require_group(group_name)
  root[group_name] = dataset


def h5_save_dataset(save_file_name,dataset_name,dataset):
    with h5py.File(save_file_name, 'a') as fout:
        fout.require_dataset(dtype=dataset.dtype,
                             compression="gzip",
                             chunks=True,
                             name=dataset_name,
                             shape=dataset.shape)
        fout[dataset_name][...] = dataset

def slicer_little_helper(sliced_dataset, coordinate_dataset, line_number):
    return sliced_dataset[
        slice(coordinate_dataset[line_number,0],    coordinate_dataset[line_number,1]),
        slice(coordinate_dataset[line_number,2],    coordinate_dataset[line_number,3]),
        slice(coordinate_dataset[line_number,4],    coordinate_dataset[line_number,5]),
            ]


def sanitize_coordinate_array(coordinate_array, fov_shape):
 sca  = np.zeros_like(coordinate_array)
 for dim_index in range(3):
  bad_voi_indices = np.argwhere(coordinate_array[:,0+2*dim_index] > coordinate_array[:,1+2*dim_index])
  coordinate_array[bad_voi_indices,1+2*dim_index] = coordinate_array[bad_voi_indices,0+2*dim_index]

  sca[:,0+dim_index*2] = np.maximum(coordinate_array[:,0+dim_index*2], 0)
  sca[:,1+dim_index*2] = np.minimum(coordinate_array[:,1+dim_index*2], fov_shape[dim_index])
 return sca

def inflate_coordinates(coordinate_array, spatial_binning_factors):
    # this function performs the opposite role to rescale_coordinates - multiplication instead of division
 new_array = np.zeros_like(coordinate_array)
 for dim_index in range(3):
  new_array[:,0+2*dim_index] = np.floor(np.multiply(coordinate_array[:,0+2*dim_index].astype('float64'), spatial_binning_factors[0+dim_index]))
  new_array[:,1+2*dim_index] =  np.ceil(np.multiply(coordinate_array[:,1+2*dim_index].astype('float64'), spatial_binning_factors[0+dim_index]))

 return new_array.astype('int64')

def get_binning_factors(default_binning_factors = np.ones((3,), dtype=np.int64) ):
    default_binning_factors = np.array(default_binning_factors,dtype=np.int64) # sanitizing input
    '''
    try:
        num_lines = float(input("Please enter the number of lines in the 4D dataset. Press enter to quit  "))
        line_binning_factor = np.divide(2048,num_lines).astype('int64')
    except ValueError:
        print('Unknown number of lines! Aborting!')
    '''
    try:
        num_columns_3d = 512  #float(input("Please enter the number of columns in the 3D segmentation mask. Press enter to quit  "))
        num_columns_4d = 2048 #float(input("Please enter the number of columns in the 4D dataset. Press enter to quit  "))
        default_binning_factors[1] = np.divide(num_columns_4d,num_columns_3d).astype('int64')
        print(f"The voxel binning factors have been set to {default_binning_factors}")
    except ValueError:
        print('Unknown number of columns! Aborting!')
    return default_binning_factors

'''
raw_coordinate_array = fiji_voi_reader(VOI_file_name, axial_feature_radius, data_shape[-1])

print(sample_file_name)
try:
 num_lines = float(input("Please enter the number of lines. Press enter to quit  "))
 line_binning_factor = np.divide(2048,num_lines).astype('int64')
except ValueError:
 print('Unknown number of lines! Aborting!')

try:
 pixel_to_micron_ratio = float(input(f"Please enter the pixel to micron ratio. The default value is {default_pixel_to_micron_ratio} pixels to micron.   "))
except ValueError:
 pixel_to_micron_ratio = default_pixel_to_micron_ratio
print(f"The pixel to micron ratio has been set to {pixel_to_micron_ratio}")

binning_factors = (line_binning_factor,1,1) # e.g. (8,1,1) if there are 256 lines in full stack files vs. 2048 lines in VOI reference dataset

unsanitized_coordinate_array = rescale_coordinates(raw_coordinate_array, binning_factors)

coordinate_array = sanitize_coordinate_array(unsanitized_coordinate_array, data_shape[1:]) # disregard time dimension when sanitizing coordinate array

coordinate_fname = VOI_file_name[:-4] + '_sanitized.zarr'

try:
  save_dataset(coordinate_fname, 'coordinate_array', coordinate_array)
except:
  print('coordinate array file found: ' + coordinate_fname)

#print(coordinate_array)
'''



'''
@jit(parallel=True, fastmath=True)
def gimme_zarr_contents(dzh, ca, cs):
  # conversion to float64 only attempts to bypass an apparent numba bug
   return np.array(dzh[(slice(0, number_of_cropped_frames), slice(ca[feature_number,0],ca[feature_number,1]), slice(ca[feature_number,2], ca[feature_number,3]), cs)]  )#.astype('float64'))

#@njit(parallel=True, fastmath=True)
@jit(parallel=True, fastmath=True) # failed to run numba_me in nopython mode
def numba_me(a,b):
  return a * np.sum(b, axis=(1,2))
'''



def calc_vascular_diameter(thresholded_radon_profile):
    # Find the vascular diameter in real space corresponding to a thresholded radon_profile in Radon space
    straight_up_radon_transform = np.zeros((thresholded_radon_profile.size,180))
    straight_up_radon_transform[:,0] = thresholded_radon_profile
    straight_up_vessel = iradon(straight_up_radon_transform)
    straight_up_vessel_quartile = np.divide(straight_up_vessel.shape[0], 4).astype('int64')
    straight_vascular_profile = np.nansum(straight_up_vessel[straight_up_vessel_quartile:-straight_up_vessel_quartile,:],axis=0) # the contrast in Radon space may be clipped towards the edge of the image

    diff_straight_vascular_profile = np.abs(np.diff(straight_vascular_profile)) # the greatest absolute difference between neighbouring pixels is expected at the edges of the radon profile
    height_threshold = np.divide(np.nanmax(diff_straight_vascular_profile),100)

    my_peaks, dict_peak = find_peaks(diff_straight_vascular_profile, height=height_threshold) 
    sort_my_dict = np.argsort(dict_peak["peak_heights"])
    #print(dict_peak["prominences"][sort_my_dict])
    #print(my_peaks[sort_my_dict])
    
    try:
      vascular_diameter = np.abs(my_peaks[sort_my_dict[-1]]-my_peaks[sort_my_dict[-2]]) # vascular diameter corresponds to the distance between the two most prominent peaks
    except:
      vascular_diameter = np.nan

    return vascular_diameter

def calc_global_radon_threshold(radon_profile):
  # Use the time-collapsed imagery to set a single threshold in Radon space, which will be then imposed on all frames over time
  radon_profile = radon_profile.astype('float') # convert to float precision to improve division
  radon_profile = radon_profile - np.nanmin(radon_profile)
  radon_profile = np.squeeze(radon_profile)
  return 0.5 * np.nanmax(radon_profile)

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

def thresholded_radon_profiler(radon_profile):
    # Return the thresholded profile in Radon space for a specific angle
    radon_profile = radon_profile.astype('float') # convert to float precision to improve division
    radon_profile = radon_profile - np.nanmin(radon_profile)
    radon_profile = np.squeeze(radon_profile)
    fwhm_threshold = 0.5 # look for values exceeding 50% of maximal brightness
    return (radon_profile > fwhm_threshold*np.nanmax(radon_profile))

def dynamic_fwhm_calc(radon_profile, global_threshold=0.5):
# Find the full width at half maximum along the vascular profile in radon space, using a time-collapsed intensity threshold 
  radon_profile = radon_profile.astype('float') 
  radon_profile = radon_profile - np.nanmin(radon_profile)
  radon_profile = np.squeeze(radon_profile)

  over_threshold_indices = np.argwhere(radon_profile > global_threshold)
  if over_threshold_indices.size > 1:
    time_varying_radon_fwhm = over_threshold_indices[-1] - over_threshold_indices[0]
    if time_varying_radon_fwhm.size > 1:
      print('Multiple diameter values found at some time frame!')
      print('These might help debugging the error:')
      print('radon_profile shape is: ' + str(radon_profile.shape))
      print('global_threshold value is: ' + str(global_threshold))
      print('Shape of over_threshold_indices is: ' + str(over_threshold_indices.shape))
      print('Multiple diameter values are: ' + str(time_varying_radon_fwhm))
  else:
    time_varying_radon_fwhm = np.nan
    print('non-positive fwhm values found!')        

  return time_varying_radon_fwhm




#@jit(parallel=True, fastmath=True) # this function calls zarr handles, so nopython mode won't work
def milkman(dask_zarr_handle, coor_array, downscale_factors, rap, radon_vascular_angles, minimal_number_of_frames):
    num_features = coor_array.shape[0]
    ###############################
    num_features = 2 # TEMPORARY!!!
    ###############################
    print(str(num_features) +  " features identified")


    time_varying_radon_diameter = np.zeros((minimal_number_of_frames,num_features))
    time_varying_diameter_in_microns = np.zeros((minimal_number_of_frames,num_features))
    radon_to_real_space_conversion_factor = np.zeros((1,num_features))
    
    photon_count_lp = np.zeros((minimal_number_of_frames,num_features))
    global_radon_threshold = np.zeros((num_features,))
  


    for feature_number in range(num_features):
        print("cropping feature number " + str(feature_number))

        num_cropped_slices = coordinate_array[feature_number,5]-coordinate_array[feature_number,4]


        slice_my_feature = (slice(0, minimal_number_of_frames), slice(coordinate_array[feature_number,0],coordinate_array[feature_number,1]), slice(coordinate_array[feature_number,2], coordinate_array[feature_number,3]), slice(coordinate_array[feature_number,4], coordinate_array[feature_number,5]))
        print(slice_my_feature)
        print(dask_zarr_handle[slice_my_feature].shape)

        downscaled_feature = downscale_local_mean(np.einsum('ijkl,l->ijk', np.array(dask_zarr_handle[slice_my_feature]), rap[slice(coordinate_array[feature_number,4], coordinate_array[feature_number,5])]), downscale_factors)
        
        sos = butter(4, lowpass_frequency, 'lp', fs=exact_frame_rate, output='sos')

    

        downscaled_feature_lp = sosfiltfilt(sos, downscaled_feature, axis=0)
        photon_count_lp[:,feature_number] = np.sum(downscaled_feature_lp, axis=(1,2))

        print('shape of downscaled_feature_lp is: '+ str(downscaled_feature_lp.shape))

        downscaled_feature_lp_mean = np.mean(downscaled_feature_lp, axis=0)
        inflated_image = rescale(downscaled_feature_lp_mean,(line_binning_factor,1))
        time_collapsed_radon_profile = radon(np.squeeze(inflated_image), theta=radon_vascular_angles[feature_number], circle=False, preserve_range=False)      
        global_radon_threshold[feature_number] = calc_global_radon_threshold(time_collapsed_radon_profile) # using the mean downscaled_feature_lp for a smoother, time-invariant radon threshold imposed on all frames

        radon_to_real_space_conversion_factor[0,feature_number] = np.divide(calc_vascular_diameter( thresholded_radon_profiler(time_collapsed_radon_profile)) , dynamic_fwhm_calc(time_collapsed_radon_profile, global_radon_threshold[feature_number]))

        inflated_movie = rescale(downscaled_feature_lp,(1,line_binning_factor,1))
        print('shape of inflated_movie is: ' + str(inflated_movie.shape))
        for frame_number in range(minimal_number_of_frames):
          momentary_radon_profile = radon(np.squeeze(inflated_movie[frame_number,:,:]), theta=radon_vascular_angles[feature_number], circle=False, preserve_range=False)
          time_varying_radon_diameter[frame_number, feature_number] = dynamic_fwhm_calc(momentary_radon_profile, global_radon_threshold[feature_number])
    time_varying_diameter_in_microns = np.divide(diameter_correction_factor*time_varying_radon_diameter*radon_to_real_space_conversion_factor, pixel_to_micron_ratio)
    print(f'radon_to_real_space_conversion_factor is {radon_to_real_space_conversion_factor}')
    return downscaled_feature_lp, time_varying_diameter_in_microns, photon_count_lp



def milk_many_files(volumetric_fname, coordinate_fname, ds_factors, rap_fname, biological_data_type):
#def milk_many_files(file_wc, coordinate_array, reciprocal_axial_intensity_profile):
        print(volumetric_fname)
        z = zarr.open(volumetric_fname,mode='r')
        save_fn = volumetric_fname[:-5] + '_' + str(biological_data_type) + '_' + str(lowpass_frequency) + '_Hz_lowpass_filtering_radon.zarr'

        #reciprocal_axial_intensity_profile = np.squeeze(np.load(rap_fname))
        reciprocal_axial_intensity_profile = np.array(zarr.open(rap_fname,mode='r'))

        cz = zarr.open(coordinate_fname,mode='r')
        coordinate_array = np.array(cz['coordinate_array'])

        save_dataset(save_fn, 'coordinate_array', coordinate_array)

        data_shape = z['Full Stack']['Channel 1'].shape
        dz =  da.from_zarr(volumetric_fname, 'Full Stack/Channel 1')

        hdf5_file_name = save_fn.replace("zarr", "hdf5")

        num_features = coordinate_array.shape[0]
        minimal_number_of_frames = np.minimum(number_of_cropped_frames, data_shape[0]) # accounting for datasets whose number of frames is smaller than number_of_cropped_frames


        vascular_angles_in_radon_space = np.zeros((num_features,1))

        for feature_number in range(num_features):
          feature_shape = np.einsum('ijk,k->ij', z['Summed Stack']['Channel 1'][(slice(coordinate_array[feature_number,0],coordinate_array[feature_number,1]), slice(coordinate_array[feature_number,2], coordinate_array[feature_number,3]), slice(coordinate_array[feature_number,4], coordinate_array[feature_number,5]))], reciprocal_axial_intensity_profile[slice(coordinate_array[feature_number,4], coordinate_array[feature_number,5])].astype('float64'))
          save_dataset(save_fn,'cropped_feature_shape_' + str(feature_number), feature_shape )

          vascular_angles_in_radon_space[feature_number] = find_vascular_angle(feature_shape)

          h5_save_dataset(hdf5_file_name,'cropped_feature_shape_' + str(feature_number),feature_shape)

        
        

        print('initiating dynamics milking')
        milking_start_time = time.time()
        downscaled_feature_lp, time_varying_diameter_in_microns, photon_count_lp = milkman(dz, coordinate_array, ds_factors, reciprocal_axial_intensity_profile, vascular_angles_in_radon_space, minimal_number_of_frames)



        print("dynamics milking takes %s seconds ---" % (time.time() - milking_start_time))



        h5_save_dataset(hdf5_file_name,'branching_vessel_side_view',downscaled_feature_lp)
        h5_save_dataset(hdf5_file_name,'branching_vessel_photon_count',photon_count_lp)
        h5_save_dataset(hdf5_file_name,'vascular_angle',vascular_angles_in_radon_space)
        h5_save_dataset(hdf5_file_name,'time_varying_vascular_diameter',time_varying_diameter_in_microns)

        save_dataset(save_fn,'branching_vessel_side_view',downscaled_feature_lp)
        save_dataset(save_fn,'branching_vessel_photon_count',photon_count_lp)
        save_dataset(save_fn,'vascular_angle',vascular_angles_in_radon_space)
        save_dataset(save_fn,'time_varying_vascular_diameter',time_varying_diameter_in_microns)
        

        




 
if __name__=='__main__':

    zarr.storage.default_compressor = Zstd(level=3)

    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)


    num_segments = 4738

    #segments_fld = os.path.join('D:' + os.sep, 'code')
    segments_fld = os.path.join('D:' + os.sep, 'Downloads')
    
    #segments_fn = os.path.join(segments_fld, 'mygraph_2020_01_13_512x512x180_segments.zarr')
    #segments_fn = os.path.join(segments_fld, 'binary_mask_100x130x180_segments.zarr')
    segments_fn = os.path.join(segments_fld, 'ref_2020_01_13_512x512x150_binary_otsu_1p0_scaling_segments.zarr')


    #segments_npy_fn = segments_fn.replace('.zarr','.npy')

    segments_zarr_handle =  zarr.open(segments_fn,mode='r')

    #segment_slice_indices = np.load(segments_npy_fn)
    segment_slice_indices = np.array(segments_zarr_handle['segment_slice_indices'])
    mask_slice_indices =    np.array(segments_zarr_handle['segment_mask_slice_indices'])



    dynamics_fld =  os.path.join('D:' + os.sep, 'Downloads')
    dynamics_fn = os.path.join(dynamics_fld, 'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_512l_1slow_3p5x_zoom_+100um_height_FOV1_034.zarr')
    print(dynamics_fn)
    dynamics_zarr_handle = zarr.open(dynamics_fn,mode='r')

    #'''
    ###Temporary###

    fld = 'D:/Downloads/'

    fn = fld + 'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_1024l_1slow_3p5x_zoom_+100um_heigh_summed_bc_t_FOV1_062_summed_bc_top180_slices_rs512x2048_for_vg.tif'
    #fn = fld + 'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_1024l_1slow_3p5x_zoom_+100um_heigh_summed_bc_t_FOV1_062_summed_bc_top180_slices_rs512x2048_for_vg_gamma2_rs_512x512x180_cropped_100x130x180.tif'
    save_fld = 'D:/code/VascularGraph/'
    sfn = save_fld + 'binary_mask.mat'

    segmentation_mask = np.array(tf.imread(fn))

    segmentation_mask = np.moveaxis(segmentation_mask, 0, -1) # FiJI turns the depth dimension into the first (0th) dimension. Here we fix it back

    print(f'shape of full segmentation_mask is {segmentation_mask.shape}')

    test_my_masks = np.zeros_like(segmentation_mask)

    #'''


    #hdf5_file_name = dynamics_fn.replace(".zarr", "_cylinderified.hdf5")
    hdf5_file_name = fn.replace(".tif", "_cylinderified.hdf5") ###Temporary###


    '''
    summed_stack_shape = np.array(dynamics_zarr_handle['Summed Stack']['Channel 1']).shape
    print(f'shape of summed_stack_shape is {summed_stack_shape}')
    '''
    
    #print(segment_slice_indices)

    segment_array = np.array([300, 538, 752, 1327, 1328, 1337, 1338, 1360, 1361, 1368, 1372, 1373, 1391, 1411, 1504, 1508, 1653, 1654, 1708, 1709, 1732, 1774, 1775,
    1797, 1836, 1909, 1914, 1915, 1959, 1960, 1962, 1975, 1999, 2002, 2008, 2011, 2012, 2048, 2049, 2059, 2064, 2097, 2102, 2111, 2127, 2131,
    2132, 2137, 2138, 2147, 2184, 2210, 2244, 2338, 2366, 2367 ,2372, 2373, 2431, 2501, 2502, 2523, 2524, 2554, 2570, 2578, 2594, 2609, 
    2615, 2620, 2637, 2642, 2673, 2676, 2700])
    



    #for segment_number in segment_array:
    #for segment_number in range(9752):
    for segment_number in range(num_segments):     
        #mask_number = np.array(segment_number + 1) # TEMPORARY
        mask_number = np.array(segment_number)
        mask_dataset_name = 'segment_'+str(mask_number)+'_mask'
        #print(f'segment_slice_indices of segment {mask_number} are {segment_slice_indices[mask_number,:]}')
        

        #sliced_summed_stack = np.array(dynamics_zarr_handle['Summed Stack']['Channel 1'][# the skeletonized graph has different order of dimensions than the full stack

        '''
        sliced_summed_stack = np.array(segmentation_mask[
            slice(segment_slice_indices[mask_number,2], segment_slice_indices[mask_number,3]),
            slice(segment_slice_indices[mask_number,4], segment_slice_indices[mask_number,5]),
            slice(segment_slice_indices[mask_number,0],segment_slice_indices[mask_number,1])
        '''
        
        '''
        sliced_summed_stack = np.array(segmentation_mask[ slice(segment_slice_indices[mask_number,0], segment_slice_indices[mask_number,1]), slice(segment_slice_indices[mask_number,2], segment_slice_indices[mask_number,3]), slice(segment_slice_indices[mask_number,4],segment_slice_indices[mask_number,5]) ])

        
        

        print(f'shape of sliced_summed_stack is {sliced_summed_stack.shape}')
        print(f'shape of segmentation mask is {np.array(segments_zarr_handle[mask_dataset_name]).shape}')
        '''
        try:
            test_my_masks[ slice(segment_slice_indices[mask_number,0], segment_slice_indices[mask_number,1]), slice(segment_slice_indices[mask_number,2], segment_slice_indices[mask_number,3]), slice(segment_slice_indices[mask_number,4],segment_slice_indices[mask_number,5]) ] += np.array(segments_zarr_handle[mask_dataset_name])
        except:
            print(f'failure with segment {segment_number}')
            print(f'segment_slice_indices are {segment_slice_indices[mask_number,:]}')
            a = test_my_masks[ slice(segment_slice_indices[mask_number,0], segment_slice_indices[mask_number,1]), slice(segment_slice_indices[mask_number,2], segment_slice_indices[mask_number,3]), slice(segment_slice_indices[mask_number,4],segment_slice_indices[mask_number,5]) ]
            print(f'shapes of sliced segment_slice_indices and boolean mask are {a.shape} and  {np.array(segments_zarr_handle[mask_dataset_name]).shape}, respectively')

        '''
        try:
            cylinderified_summed_stack = np.einsum('ijk,ijk->ijk',sliced_summed_stack,np.array(segments_zarr_handle[mask_dataset_name]))
            print('ijk worked')
            #cylinderified_summed_stack = np.einsum('ikj,ijk->ijk',sliced_summed_stack,np.array(segments_zarr_handle[mask_dataset_name]))

            #cylinderified_summed_stack = np.einsum('ijk,ikj->ijk',sliced_summed_stack,np.array(segments_zarr_handle[mask_dataset_name]))
            #cylinderified_summed_stack = np.einsum('ijk,ijk->ijk',sliced_summed_stack,np.moveaxis(np.array(segments_zarr_handle[mask_dataset_name]),[0,1,2],[1,2,0]))            
        except:
            try:
                print('0 1 2 failed')

                sliced_summed_stack = np.array(segmentation_mask[ slice(segment_slice_indices[mask_number,0]+47, segment_slice_indices[mask_number,1]+47), slice(segment_slice_indices[mask_number,2]+47, segment_slice_indices[mask_number,3]+47), slice(segment_slice_indices[mask_number,4]+47,segment_slice_indices[mask_number,5]+47) ])
                cylinderified_summed_stack = np.einsum('ijk,ijk->ijk',sliced_summed_stack,np.array(segments_zarr_handle[mask_dataset_name]))

                #cylinderified_summed_stack = np.einsum('jki,ijk->ijk',sliced_summed_stack,np.array(segments_zarr_handle[mask_dataset_name]))
            
                #cylinderified_summed_stack = np.einsum('ijk,jki->ijk',sliced_summed_stack,np.array(segments_zarr_handle[mask_dataset_name]))
                #cylinderified_summed_stack = np.einsum('ijk,ijk->ijk',sliced_summed_stack,np.moveaxis(np.array(segments_zarr_handle[mask_dataset_name]),[0,1,2],[0,2,1]))
            except:
                cylinderified_summed_stack = np.array(([1,1],[1,1],[1,1]))
                print('both options failed')

        '''                
        '''
        # This code block was used to reveal how the axes of the skeletonized mask should be moved to suit those of the summed stack:

        

        mm = np.array(segments_zarr_handle[mask_dataset_name])
        print(f'mask shape is {mm.shape}')

        print(sliced_summed_stack.shape)
       
        try:
            cylinderified_summed_stack = np.einsum('ijk,ijk->ijk',sliced_summed_stack,np.moveaxis(np.array(segments_zarr_handle[mask_dataset_name]),[0,1,2],[1,2,0]))
            cylinderified_summed_stack = np.einsum('ijk,ijk->ijk',sliced_summed_stack,np.moveaxis(np.array(segments_zarr_handle[mask_dataset_name]),[0,1,2],[0,2,1]))
            print('both options work')
        except:
            try:
                cylinderified_summed_stack = np.einsum('ijk,ijk->ijk',sliced_summed_stack,np.moveaxis(np.array(segments_zarr_handle[mask_dataset_name]),[0,1,2],[0,2,1]))
                print('0 2 1')
            except:
                try:
                    cylinderified_summed_stack = np.einsum('ijk,ijk->ijk',sliced_summed_stack,np.moveaxis(np.array(segments_zarr_handle[mask_dataset_name]),[0,1,2],[1,2,0]))
                    print('1 2 0')
                except:
                    print('both options failed')
        '''


        #h5_save_dataset(hdf5_file_name,'segment_'+str(segment_number)+'_before',sliced_summed_stack)
        #h5_save_dataset(hdf5_file_name,'segment_'+str(segment_number)+'_after',cylinderified_summed_stack)
    biological_data_type = 'vascular' # Are you milking vessel segments, neuronal cell bodies, or both?


    #all_files = [(fname, coordinate_fname, downscale_factors, axial_intensity_file_name, biological_data_type) for fname in glob.glob(file_wild_card)]
    #print(str(len(all_files)) + ' files found')



'''
    with multiprocessing.Pool(processes=4) as mp:
        mp.starmap(milk_many_files, all_files)
        
'''

h5_save_dataset(hdf5_file_name,'test_my_masks',test_my_masks)