# Milk temporal dynamics in previously segemented volumes of interest, preparing them for downstream analysis
# source activate opencv



#import cv2
import csv
from numba import jit, njit
import numpy as np
import zarr
from numcodecs import Blosc, Zstd
import time
import dask.array as da
import dask

zarr.storage.default_compressor = Zstd(level=3)

compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

# Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing



data_folder_name = '/pblab/pblab/Lior/control/'


save_folder_name = data_folder_name

#data_file_name = 'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_256l_1slow_3p5x_zoom_+100um_height_max_power_FOV1_VISIBLE_NEURONS_326.zarr'
#data_file_name = 'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_512l_1slow_3p5x_zoom_+100um_height_FOV1_034.zarr'
#data_file_name = 'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_1024l_1slow_3p5x_zoom_+100um_height_FOV1_061.zarr'
#data_file_name = 'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_256l_1slow_3p5x_zoom_+100um_height_FOV1_169.zarr'
data_file_name = 'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_256l_0p5slow_3p5x_zoom_+100um_height_max_power_FOV1_257.zarr'

axial_intensity_file_name = 'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_256l_0p5slow_3p5x_zoom_+100um_height_max_powe_summed_bc_r_FOV1_257_rap.zarr'
VOI_file_name = 'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_256l_1slow_3p5x_zoom_+100um_height_max_power_FOV1_VISIBLE_N_summed_bc_EURONS_327_summed_bc_rs_2048x2048_3d_gauss_3x3x1_squared_Overlay_Elements_CONTROL_SET.csv'


axial_feature_radius = int(12) # grabbing 12 slices above and below the center of each feature of interest



sample_file_name = data_folder_name + data_file_name

save_fn = save_folder_name + 'CONTROL_' + data_file_name[:-5] + '_neu_traces_tz.zarr'

z = zarr.open(sample_file_name)
print(z.array_keys())


# full_stack_zarr = z['Full Stack']['Channel 1']


data_shape = z['Full Stack']['Channel 1'].shape
#dz =  da.from_zarr(sample_file_name,'Full Stack/Channel 1', chunks=(100, 6, 6, 1))
dz =  da.from_zarr(sample_file_name,'Full Stack/Channel 1')


number_of_cropped_frames = data_shape[0]

reciprocal_axial_intensity_profile = np.array(zarr.open(data_folder_name + axial_intensity_file_name,mode='r'))

def save_dataset(save_file_name,group_name,dataset):
  root = zarr.open_group(save_file_name, mode='a')
  fill_me = root.require_group(group_name)
  root[group_name] = dataset
#  compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
#  zarr.save_group(save_file_name, group_name=dataset, compressor=compressor, chunks=True)
  #zarr.save_group(save_file_name, group_name=dataset, chunks=True)


def fiji_voi_reader(file_name, axial_feature_radius, num_slices):
 coordinate_array = np.zeros((0,6)).astype('int64')
 specific_voi_coordinates = np.zeros((6,1)).astype('int64')

 with open(file_name, newline='') as f:
  reader = csv.reader(f)
  for counter, row in enumerate(reader):
   if counter == 0:
    continue
   print('counter = ' + str(counter))
   specific_voi_coordinates[0] = int(row[4]) - 1 # FIJI indices run from 1 rather than 0
   specific_voi_coordinates[1] = specific_voi_coordinates[0] + int(row[6])
   specific_voi_coordinates[2] = int(row[3]) - 1 # FIJI indices run from 1 rather than 0
   specific_voi_coordinates[3] = specific_voi_coordinates[2] + int(row[5])
   specific_voi_coordinates[4] = np.maximum(int(row[-2]) - 1 - int(axial_feature_radius), 0)
   specific_voi_coordinates[5] = np.minimum(int(row[-2]) - 1 + int(axial_feature_radius), num_slices)

   coordinate_array = np.append(coordinate_array,specific_voi_coordinates.T,axis=0)

 return coordinate_array


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

sample_file_name = data_folder_name + data_file_name



raw_coordinate_array = fiji_voi_reader(data_folder_name+VOI_file_name, axial_feature_radius, data_shape[-1])

print(sample_file_name)
try:
 #num_lines = float(input("Please enter the number of lines. Press enter to quit  "))
 num_lines = 256
 line_binning_factor = int(np.round(2048/num_lines))
except ValueError:
 print('Unknown number of lines! Aborting!')

binning_factors = (line_binning_factor,1,1) # e.g. (8,1,1) if there are 256 lines in full stack files vs. 2048 lines in VOI reference dataset

unsanitized_coordinate_array = rescale_coordinates(raw_coordinate_array, binning_factors)

coordinate_array = sanitize_coordinate_array(unsanitized_coordinate_array, data_shape[1:]) # disregard time dimension when sanitizing coordinate array

#print(coordinate_array)

save_dataset(save_fn, 'coordinate_array', coordinate_array)

'''
num_features = coordinate_array.shape[0]
print(str(num_features) +  " features identified")

feature_dynamics_tz =  np.zeros((num_features,number_of_cropped_frames, int(np.ceil(axial_feature_radius*2+1)))) #
feature_dynamics = np.zeros((num_features,number_of_cropped_frames)) #
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


@jit(parallel=True, fastmath=True) # this function calls zarr handles, so nopython mode won't work
def milkman(dask_zarr_handle, coor_array, rap, num_frames):
  num_features = coor_array.shape[0]
  print(str(num_features) +  " features identified")
  feature_dynamics_tzc =  np.zeros( (number_of_cropped_frames, int(np.ceil(axial_feature_radius*2+1)), num_features) ) #
  for feature_number in range(num_features):

    print("cropping feature number " + str(feature_number))

    num_cropped_slices = coordinate_array[feature_number,5]-coordinate_array[feature_number,4]

    #for slice_num, current_slice in enumerate(range(coordinate_array[feature_number,5]), coordinate_array[feature_number,4]):
    #  #current_slice = coordinate_array[feature_number,4] + slice_num
    #  feature_dynamics_tzc[:,slice_num,feature_number] = numba_me(rap[current_slice].astype('float64') , gimme_zarr_contents(dask_zarr_handle, coor_array, current_slice))

    slice_my_feature = (slice(0, number_of_cropped_frames), slice(coordinate_array[feature_number,0],coordinate_array[feature_number,1]), slice(coordinate_array[feature_number,2], coordinate_array[feature_number,3]), slice(coordinate_array[feature_number,4], coordinate_array[feature_number,5]))
    feature_dynamics_tzc[:,:num_cropped_slices,feature_number] = np.einsum('ijkl,l->il', np.array(dask_zarr_handle[slice_my_feature]), rap[slice(coordinate_array[feature_number,4], coordinate_array[feature_number,5])])
  return feature_dynamics_tzc

for feature_number in range(coordinate_array.shape[0]):
    feature_shape = np.einsum('ijk,k->ijk', z['Summed Stack']['Channel 1'][(slice(coordinate_array[feature_number,0],coordinate_array[feature_number,1]), slice(coordinate_array[feature_number,2], coordinate_array[feature_number,3]), slice(coordinate_array[feature_number,4], coordinate_array[feature_number,5]))], reciprocal_axial_intensity_profile[slice(coordinate_array[feature_number,4], coordinate_array[feature_number,5])].astype('float64'))
    save_dataset(save_fn,'cropped_feature_shape_' + str(feature_number), feature_shape )

print('initiating dynamics milking')
milking_start_time = time.time()
feature_dynamics_tz = milkman(dz, coordinate_array, reciprocal_axial_intensity_profile, 10)
print("dynamics milking takes %s seconds ---" % (time.time() - milking_start_time))

save_dataset(save_fn,'neuronal_dynamics_tz_' + str(number_of_cropped_frames), feature_dynamics_tz)
save_dataset(save_fn,'neuronal_dynamics_' + str(number_of_cropped_frames), np.sum(feature_dynamics_tz,axis=1))


