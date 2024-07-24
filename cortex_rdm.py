# Milk temporal dynamics from penetrating arteries in previously segemented volumes of interest, preparing them for downstream analysis
# conda activate pysight



#import cv2
import csv
from numba import jit, njit
import numpy as np
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

from skimage.transform import downscale_local_mean, rescale


from skimage.transform import radon, iradon


import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.stats import variation

zarr.storage.default_compressor = Zstd(level=3)

compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

# Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

downscale_factors = (1,1,1)
#downscale_factors = (258,1,1,1)

number_of_cropped_frames = 495 # np.minimum(900, int(data_shape[0])) # 3000 # int(data_shape[0]) # 25

exact_frame_rate = 30.02 # volumes per second in original recording
lowpass_frequency = 1 # [Hz]

data_folder_name = r'/data/Lior/lst/tt/2021_02_01/' # '/data/Lior/lst/2020/2020_01_13/'
default_pixel_to_micron_ratio = float(512/500) # float(2048/543)
diameter_correction_factor = 1.2 # Vascular diameter is typically 20% wider than FWHM thresholding determines

save_folder_name = data_folder_name


file_wild_card = data_folder_name + '*FOV3*00[0-9].zarr'
flipped_channel_flag = True # set to True if the vascular channel is Channel 2. The default vascular channel is Channel 1
#file_wild_card = data_folder_name + 'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_512l_1slow_3p5x_zoom_+100um_height_FOV1_???.zarr'

axial_intensity_file_name = data_folder_name + '2021_02_01_19_19_39_mouse_flipped_channels__1_FOV3_6100um_deep_512l_3x_mag_1850_sec_acq_with_FLIM_4Mcps_laser_pulses_with_TAG_00053_summed_bc.zarr'
#axial_intensity_file_name = data_folder_name + 'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_256l_0p5slow_3p5x_zoom_+100um_height_max_powe_summed_bc_r_FOV1_257_rap.zarr'
VOI_file_name = data_folder_name + 'Overlay_Elements_of_2021_02_01_19_19_39_mouse_flipped_channels__1_FOV3_6100um_deep_512l_3x_mag_1850_sec_acq_with_FLIM_4Mcps_laser_pulses_with_TAG_00053_summed_bc_improved_vascular_contrast_equalized.csv'
#'Overlay Elements of TLP1_Thy2_GCaMP7_FITC_mouse_930nm_1024l_1slow_3p5x_zoom_+100um_heigh_summed_bc_t_FOV1_062_summed_bc_top256_slices_rs2048x2048_bc_RoiSet_branching_vessels_only.csv'

axial_feature_radius = int(3) # int(20) # grabbing 20 slices above and below the center of each feature of interest



fn_list = glob.glob(file_wild_card)
sample_file_name = fn_list[0]
z = zarr.open(sample_file_name,mode='r')


print(z.array_keys())




data_shape = z['Full Stack']['Channel 1'].shape

print(f'data shape[0] is: {data_shape[0]}') # debugging






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




def fiji_voi_reader(file_name, axial_feature_radius, num_slices):
  df = pd.read_csv(file_name)
  voi_coordinates = np.zeros((df.shape[0],6),dtype='int64')
  voi_coordinates[:,0] = np.array(df['Y']) - 1 # FiJI indices run from 1 rather than 0
  voi_coordinates[:,1] = voi_coordinates[:,0] + np.array(df['Height'])
  voi_coordinates[:,2] = np.array(df['X']) - 1 # FiJI indices run from 1 rather than 0
  voi_coordinates[:,3] = voi_coordinates[:,2] + np.array(df['Width'])
  voi_coordinates[:,4] = np.maximum(np.array(df['Z']) - 1 - int(axial_feature_radius) , 0)
  voi_coordinates[:,5] = np.minimum(np.array(df['Z']) - 1 + int(axial_feature_radius) , num_slices)
  print(voi_coordinates.shape)
  return voi_coordinates




def sanitize_coordinate_array(coordinate_array, fov_shape):
 sca  = np.zeros_like(coordinate_array)
 for dim_index in range(3):
  bad_voi_indices = np.where(coordinate_array[:,0+2*dim_index] > coordinate_array[:,1+2*dim_index])
  coordinate_array[bad_voi_indices,1+2*dim_index] = coordinate_array[bad_voi_indices,0+2*dim_index]

  sca[:,0+dim_index*2] = np.maximum(coordinate_array[:,0+dim_index*2], 0)
  sca[:,1+dim_index*2] = np.minimum(coordinate_array[:,1+dim_index*2], fov_shape[dim_index])
 return sca

def rescale_coordinates(coordinate_array, spatial_binning_factors):
 new_array = np.zeros_like(coordinate_array)
 for dim_index in range(3):
  new_array[:,0+2*dim_index] = np.floor(np.divide(coordinate_array[:,0+2*dim_index].astype('float64'), spatial_binning_factors[0+dim_index]))
  new_array[:,1+2*dim_index] =  np.ceil(np.divide(coordinate_array[:,1+2*dim_index].astype('float64'), spatial_binning_factors[0+dim_index]))

 return new_array.astype('int64')
###

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
    #num_features = 2 # TEMPORARY!!!
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
        vascular_channel_string = 'Channel 1'
        rap_dataset_string = 'reciprocal_axial_intensity_profile_ch1'

        if(flipped_channel_flag):
          vascular_channel_string = vascular_channel_string.replace('1','2')
          rap_dataset_string = rap_dataset_string.replace('1','2')

        z = zarr.open(volumetric_fname,mode='r')
        save_fn = volumetric_fname[:-5] + '_' + str(biological_data_type) + '_' + str(lowpass_frequency) + '_Hz_lowpass_filtering_radon.zarr'

        #reciprocal_axial_intensity_profile = np.squeeze(np.load(rap_fname))
        rap_z = zarr.open(rap_fname, mode='r')
        reciprocal_axial_intensity_profile = np.array(rap_z[rap_dataset_string])

        cz = zarr.open(coordinate_fname,mode='r')
        coordinate_array = np.array(cz['coordinate_array'])

        save_dataset(save_fn, 'coordinate_array', coordinate_array)

        data_shape = z['Full Stack'][vascular_channel_string].shape
        dz =  da.from_zarr(volumetric_fname, 'Full Stack/'+vascular_channel_string)

        hdf5_file_name = save_fn.replace("zarr", "hdf5")

        num_features = coordinate_array.shape[0]
        minimal_number_of_frames = np.minimum(number_of_cropped_frames, data_shape[0]) # accounting for datasets whose number of frames is smaller than number_of_cropped_frames


        vascular_angles_in_radon_space = np.zeros((num_features,1))

        for feature_number in range(num_features):
          feature_shape = np.einsum('ijk,k->ij', z['Summed Stack'][vascular_channel_string][(slice(coordinate_array[feature_number,0],coordinate_array[feature_number,1]), slice(coordinate_array[feature_number,2], coordinate_array[feature_number,3]), slice(coordinate_array[feature_number,4], coordinate_array[feature_number,5]))], reciprocal_axial_intensity_profile[slice(coordinate_array[feature_number,4], coordinate_array[feature_number,5])].astype('float64'))
          save_dataset(save_fn,'cropped_feature_shape_' + str(feature_number), feature_shape )
          feature_in_radon_space = radon(feature_shape)

          vascular_angles_in_radon_space[feature_number] = find_vascular_angle(feature_in_radon_space)

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
        

        


biological_data_type = 'vascular' # Are you milking vessel segments, neuronal cell bodies, or both?
all_files = [(fname, coordinate_fname, downscale_factors, axial_intensity_file_name, biological_data_type) for fname in glob.glob(file_wild_card)]
print(str(len(all_files)) + ' files found')




with multiprocessing.Pool(processes=4) as mp:
    mp.starmap(milk_many_files, all_files)

#milk_many_files(file_wild_card, coordinate_fname, downscale_factors, axial_intensity_file_name, biological_data_type)
