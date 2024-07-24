# The proper environment is skimage

import tifffile
import pathlib
import h5py
import numpy as np
from skimage.transform import downscale_local_mean,rescale, resize
import csv
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as anim
from imageio_ffmpeg import write_frames
import calcium_bflow_analysis.dff_analysis_and_plotting.dff_analysis as dff_analysis
from tqdm import tqdm



'''
def save_dataset(dataset_name,dataset):
    with h5py.File(save_file_name, 'a') as fout:
        fout.require_dataset(dtype=dataset.dtype,
                             compression="gzip",
                             chunks=True,
                             name=dataset_name,
                             shape=dataset.shape)
        fout[dataset_name][...] = dataset
'''

def h5_save_dataset(save_file_name,dataset_name,dataset):
    with h5py.File(save_file_name, 'a') as fout:
        fout.require_dataset(dtype=dataset.dtype,
                             compression="gzip",
                             chunks=True,
                             name=dataset_name,
                             shape=dataset.shape)
        fout[dataset_name][...] = dataset



def save_dataset(save_file_name,group_name,dataset):
  root = zarr.open_group(save_file_name, mode='a')
  fill_me = root.require_group(group_name)
  root[group_name] = dataset


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





def rescale_coordinates(coordinate_array, spatial_binning_factors):
 new_array = np.zeros_like(coordinate_array)
 for dim_index in range(3):
  new_array[:,0+2*dim_index] = np.floor(np.divide(coordinate_array[:,0+2*dim_index].astype('float64'), spatial_binning_factors[0+dim_index]))
  new_array[:,1+2*dim_index] =  np.ceil(np.divide(coordinate_array[:,1+2*dim_index].astype('float64'), spatial_binning_factors[0+dim_index]))

 return new_array.astype('int32')

def shift_coordinates(coordinate_array, subregion_slice):
 new_array = np.zeros_like(coordinate_array)
 new_array[:,0] = coordinate_array[:,0] - subregion_slice[0]
 new_array[:,1] = coordinate_array[:,1] - subregion_slice[0]
 new_array[:,2] = coordinate_array[:,2] - subregion_slice[2]
 new_array[:,3] = coordinate_array[:,3] - subregion_slice[2]
 new_array[:,4] = coordinate_array[:,4]
 new_array[:,5] = coordinate_array[:,5]
 return new_array

def sanitize_coordinate_array(coordinate_array, fov_shape):
 sca  = np.zeros_like(coordinate_array)
 for dim_index in range(3):
  bad_voi_indices = np.where(coordinate_array[:,0+2*dim_index] > coordinate_array[:,1+2*dim_index])
  coordinate_array[bad_voi_indices,1+2*dim_index] = coordinate_array[bad_voi_indices,0+2*dim_index]

  sca[:,0+dim_index*2] = np.maximum(coordinate_array[:,0+dim_index*2], 0)
  sca[:,1+dim_index*2] = np.minimum(coordinate_array[:,1+dim_index*2], fov_shape[dim_index])
 return sca


def ellipsoid_mask(coordinate_array, feature_number, fov_shape):
    # FOV_limits = np.array(FullMonty[OdorName].shape[1:])
    fov_limits = np.array(fov_shape)
    ellipsoid_centrum_indices = np.array((np.round((coordinate_array[feature_number,0] + coordinate_array[feature_number,1])/ 2).astype("int64"),
                                   np.round((coordinate_array[feature_number,2] + coordinate_array[feature_number,3])/ 2).astype("int64"),
                                    np.round((coordinate_array[feature_number,4] + coordinate_array[feature_number,5])/ 2).astype("int64") ))

    ellipsoid_diameters = np.array((np.round((coordinate_array[feature_number,1] - coordinate_array[feature_number,0])).astype("int64"),
                                    np.round((coordinate_array[feature_number,3] - coordinate_array[feature_number,2])).astype("int64"),
                                    np.round((coordinate_array[feature_number,5] - coordinate_array[feature_number,4])).astype("int64") ))

    ogrid_thingy = np.ogrid[0:fov_limits[0], 0:fov_limits[1], 0:fov_limits[2]]

    centralized_ogrid_thingy = ogrid_thingy - ellipsoid_centrum_indices

    elliptical_mask_values = np.divide((centralized_ogrid_thingy[0] * centralized_ogrid_thingy[0]),
                                     ellipsoid_diameters[0] ** 2) + \
                           np.divide((centralized_ogrid_thingy[1] * centralized_ogrid_thingy[1]),
                                     ellipsoid_diameters[1] ** 2) + \
                           np.divide((centralized_ogrid_thingy[2] * centralized_ogrid_thingy[2]),
                                     ellipsoid_diameters[2] ** 2)
    maximally_relevant_elliptical_mask_value = 0.16

    #return (elliptical_mask_values < maximally_relevant_elliptical_mask_value)
    return np.einsum('ijk,ijk->ijk', (2/maximally_relevant_elliptical_mask_value)*(maximally_relevant_elliptical_mask_value-elliptical_mask_values), (elliptical_mask_values < maximally_relevant_elliptical_mask_value))

def depth_to_color(dataset, axis=-1, colormap='plasma', slice_flag = 'False', total_num_slices = int(108), slice_range = slice(0,-1)): # encode a given axis using a color map)
 if(slice_flag):
  x = np.linspace(0.0, 1.0, total_num_slices+1)
  rgba = np.transpose(np.flip(cm.get_cmap(colormap)(x), axis=0)) # assign bright yellow values for top slices, transpose in preparation for dot product
  #print(str(slice_range))
  rgba = rgba[(slice(0,4), slice_range)]
  #print(rgba.shape)
 else:
  x = np.linspace(0.0, 1.0, np.shape(dataset)[axis])
  rgba = np.transpose(np.flip(cm.get_cmap(colormap)(x), axis=0)) # assign bright yellow values for top slices, transpose in preparation for dot product
 rgba[-1,:] = 0.5 # setting alpha value to constantly half-transparent

 return np.dot(rgba, np.moveaxis(dataset, axis, -2) ) # using np.dot rather than np.einsum to accomodate any number of dataset dimensions


def paste_neu_dynamics(feature_number):
 tc_feature_slice = (slice(small_coordinate_array[feature_number,0],small_coordinate_array[feature_number,1]), slice(small_coordinate_array[feature_number,2], small_coordinate_array[feature_number,3]), slice(small_coordinate_array[feature_number,4], small_coordinate_array[feature_number,5]))
 colored_feature = depth_to_color(small_stack[tc_feature_slice], slice_flag = 'True', total_num_slices = small_stack.shape[2], slice_range = slice(small_coordinate_array[feature_number,4], small_coordinate_array[feature_number,5]) )
 return np.multiply.outer(binned_spike_times[feature_number,:], colored_feature)

def elliptical_neu_dynamics(feature_number):
 masked_small_stack = np.einsum( 'ijk,ijk->ijk', ellipsoid_mask(small_coordinate_array, feature_number, small_stack.shape), small_stack)
 colored_feature = depth_to_color(masked_small_stack, slice_flag = 'False', total_num_slices = masked_small_stack.shape[2])
 return np.multiply.outer(binned_spike_times[feature_number,:], colored_feature)
 

def png_saver(dataset, time_axis=0, color_axis=-1):

 dataset = np.moveaxis(dataset,time_axis,0)# iterating over the time axis
 dataset = np.moveaxis(dataset,color_axis,-1) #moving color channels to the end 
 label_location = (dataset.shape[1]*0.47, dataset.shape[2]*0.07) # location of text label
 
 fig, ax = plt.subplots()
 for frame_number, frame in enumerate(dataset):   
   time_stamp = 't = ' + str(np.round(frame_number / (sampling_rate/temporal_binning_factor[1]), decimals=1)) + ' s'
   frame_file_name = save_folder_name + '/bloops/frame_' + str(frame_number) + '.png'

   plt.imshow(frame)
   if (frame_number + 1 < dataset.shape[0]): # if this is not the last frame
    plt.imshow(dataset[frame_number+1,:]) # overlaying the following frame
   ann = plt.annotate(time_stamp, label_location)
   plt.axis('off')
   plt.savefig(frame_file_name, format='png', transparent=True, dpi=300)
   ann.remove()
   plt.clf()
 return()
 
def trace_saver(dff_trace, spike_times, overhead_length = int(252), name_tag = 'dff_'):
 time_vec = np.linspace(0, 520.66417732581234607134344959085, num=65536)
 fig, ax = plt.subplots()
 for frame_number in range(full_fov.shape[0]):
   trace_file_name = save_folder_name + '/bloops/trace_' + name_tag + str(frame_number) + '.png'
   time_slice = slice(frame_number*temporal_binning_factor[1], frame_number*temporal_binning_factor[1]+skip_last_k_frames)
   ax, num_displayed_cells = dff_analysis.scatter_spikes(0.5*dff_trace[:,time_slice], spike_data=spike_times[:,time_slice], downsample_display=5, time_vec=time_vec[time_slice], ax=None)
   if (dff_trace.shape[0] == 1): # if presenting the summed stack
    ax.suptitle('logarithm of dF/F product + summed spike times')
   else:
    ax.suptitle('dF/F + spike times')
   plt.savefig(trace_file_name, format='png', transparent=True, dpi=300)
   plt.clf()
 return()

def label_me_movie(dataset, frame_rate, time_axis=0, color_axis=-1, speedup_factor=int(1)):
 dataset = np.moveaxis(dataset,time_axis,0)# iterating over the time axis
 dataset = np.moveaxis(dataset,color_axis,-1) #moving color channels to the end
 #dataset = np.nansum(dataset, axis=-1) # checking if encoding problem stems from the color channel - NO IT DOES NOT

 #dataset = dataset[:,:,:,0:3] # FileMovieWriter handles RGB but not RGBA data
 label_location = (dataset.shape[1]*0.47, dataset.shape[2]*0.07) # location of text label
 

## Writer = anim.writers['ffmpeg']
  #Writer = anim.FileMovieWriter()
## writer = Writer(fps=int(frame_rate*speedup_factor), codec='ffv1')
  #writer = Writer(fps=int(frame_rate*speedup_factor), codec='mpeg4')
  #writer = Writer(fps=frame_rate*speedup_factor, metadata=dict(artist='Me'), bitrate=1800)
  #writer = Writer(fps=frame_rate*speedup_factor)


 fig, ax = plt.subplots()

 #video_file_name = save_file_name[:-5] + '_' + str(num_cropped_frames) + '_' + str(frame_rate) + '_' + str(speedup_factor)  + '.burgul'
 video_file_name = save_folder_name + 'test7.avi'

## with writer.saving(fig, video_file_name,100):
##  for frame_number, frame in enumerate(dataset):
   #time_stamp = f't = {str(np.round(frame_number / frame_rate, decimals=1)} seconds'
   #time_stamp = 't = ' + str(np.round(frame_number / frame_rate, decimals=1)) + ' seconds'
   #print('frame shape is ' + str(frame.shape)) 
   #ax.imshow(frame)
   #ax.annotate(time_stamp, label_location)
   
   #writer.grab_frame(facecolor='k')
   #writer.grab_frame()

 # Write a video file
# writer = write_frames(video_file_name, (dataset.shape[1],dataset.shape[2]))  # size is (width, height)
 writer = write_frames(video_file_name, (512,512))  # size is (width, height)

 writer.send(None)  # seed the generator
 print('number of frames is: ' + str(dataset.shape[0]))
 #faulty_frames = [156,186,187,dataset.shape[0]-1]
 faulty_frames = [dataset.shape[0]]
 for frame_number, frame in enumerate(dataset):
    if (np.isin(frame_number,faulty_frames)):
     #continue
     writer.close()
    time_stamp = 't = ' + str(np.round(frame_number / frame_rate, decimals=1)) + ' s'
    ax.imshow(np.squeeze(np.nan_to_num(np.squeeze(frame))),shape=(512,512))
    ax.annotate(time_stamp, label_location)
    ax.axis('off')
    plt.show()
    frame = frame.copy(order='C')
    print(str(frame_number))
    try:
     writer.send(np.squeeze(np.nan_to_num(frame)))
    except:
     print('writing failed for frame ' + str(frame_number))
     ax.axis('on')
     plt.show()    
 writer.close()  # don't forget this

 return()




def movie_for_fiji(time_lapse_movie, colored_small_stack):
    tlm = np.nan_to_num(time_lapse_movie[:,slice(3),:,:].astype('float32'))
    ff = tlm + np.expand_dims(colored_small_stack[slice(3),:,:].astype('float32'), axis=0)
    #tlm = np.divide(tlm, np.nanmean(tlm, axis=(0,2,3), keepdims=True)) # normalizing each color channel to stretch along the [0,1] range
    #ff = np.divide(ff, np.nanmean(ff, axis=(0,2,3), keepdims=True)) # normalizing each color channel to stretch along the [0,1] range

    #tlm = np.nan_to_num(tlm)
    ff = np.nan_to_num(ff)
    
    # exchanging red and green channels such that superficial blood vessels will look red:
    red_channel = ff[:,0,:,:].copy() 
    ff[:,0,:,:] = ff[:,1,:,:] 
    ff[:,1,:,:] = red_channel

    # save_dataset('time_lapse_depth_colored_neurons_' + str(num_cropped_frames), tlm)
    h5_save_dataset(save_file_name, 'time_lapse_depth_colored_volume_' + str(num_cropped_frames), ff)
    #save_dataset('time_lapse_depth_colored_volume_' + str(num_cropped_frames), ff)
    return()

if __name__=='__main__':

    data_folder_name = r'/data/Lior/lst/2020/2020_02_03/FOV1/'


    save_folder_name = data_folder_name

    dff_str = 'LPT_Mouse_Thy1GCaMP6f_FITC_850mV_FOV3_2xZ_110lines_125fps_1slow_028_dataframed_LARGE_neu_traces_tz_dff.hdf5'

    tiff_str = 'LPT_Mouse_Thy1GCaMP6f_FITC_850mV_FOV3_2xZ_110lines_125fps_1slow_028_dataframed_LARGE_rescaled_brightness_corrected.tif'
    VOI_str = 'voi_list_1320_028.csv'

    save_file_name = save_folder_name + tiff_str[:-34] + '_spikes_elliptical.hdf5'

    temporal_binning_factor = (int(1),int(4))
    spatial_binning_factors = tuple(np.array([3,3,1],dtype='int64'))
    spike_highlighting_factor = int(7)
    sampling_rate = 30.02 # [volumes per second]
    num_cropped_frames = int(sampling_rate*10) # 61
    skip_first_k_frames = int(sampling_rate*0.5)
    skip_last_k_frames = int(sampling_rate*2)

    print("loading tiffstack...")
    IC_stack = np.moveaxis(tifffile.imread(data_folder_name + tiff_str), 0, 2) # reordering tiff file readout from zyx to xyz
    small_stack = downscale_local_mean(IC_stack, spatial_binning_factors, cval=0, clip=True)
    #subregion_slice = (slice(137,814),slice(677,1228),slice(0,-1))
    #small_stack2 = IC_stack[subregion_slice]

    print('shape of intensity_corrected_stack is: ' + str(IC_stack.shape))
    print('shape of downscaled stack is: ' + str(small_stack.shape))
    dynamics_ImageH5f = h5py.File(data_folder_name + dff_str, 'r')

    full_spike_times = dynamics_ImageH5f['spike_times'][:,skip_first_k_frames:num_cropped_frames+skip_first_k_frames+skip_last_k_frames]
    full_dff_trace = dynamics_ImageH5f['neuronal_dff'][:,skip_first_k_frames:num_cropped_frames+skip_first_k_frames+skip_last_k_frames]
    full_spike_times[np.isnan(full_spike_times)] = 0 # do not convert inf values to large values, so don't call nan_to_num
    full_dff_trace[np.isnan(full_dff_trace)] = 0 # do not convert inf values to large values, so don't call nan_to_num
    full_spike_times[np.isinf(full_spike_times)] = 0 # do not convert inf values to large values, so don't call nan_to_num
    full_dff_trace[np.isinf(full_dff_trace)] = 0 # do not convert inf values to large values, so don't call nan_to_num


    #normalizing brightness to arrive at bright neurons:

    full_dff_trace = np.divide(full_dff_trace, np.nanmean(full_dff_trace, axis=1, keepdims=True))
    #full_spike_times = full_spike_times + 0.5*full_dff_trace # emphasizing spikes with respect to fluorescence fluctuations


    print(full_spike_times.shape)
    print(temporal_binning_factor[1])
    #binned_spike_times = spike_highlighting_factor * temporal_binning_factor[1] * downscale_local_mean(full_spike_times[:,slice(num_cropped_frames)] + 0.2*full_dff_trace[:,slice(num_cropped_frames)], temporal_binning_factor, cval=0, clip=True)
    binned_spike_times = spike_highlighting_factor * temporal_binning_factor[1] * downscale_local_mean(full_spike_times[:,slice(num_cropped_frames)], temporal_binning_factor, cval=0, clip=True)


    num_binned_frames = binned_spike_times.shape[1]


    # for voi_fname in VOI_fnames:
    voi_fname = data_folder_name + VOI_str
    if (voi_fname):
        large_coordinate_array = fiji_voi_reader(voi_fname, 6, IC_stack.shape[-1])
        small_coordinate_array = rescale_coordinates(large_coordinate_array, spatial_binning_factors)
        small_coordinate_array = sanitize_coordinate_array(small_coordinate_array, small_stack.shape)

        print('exemplary large coordinates are ' + str(large_coordinate_array[50:52,:]))
        print('respective small coordinates are ' + str(small_coordinate_array[50:52,:]))

        #fdm = np.multiply.outer(np.ones([num_binned_frames]), small_stack) # 4D movie stub
        #fdm = np.repeat(np.expand_dims(np.zeros_like(small_stack), axis=0), num_binned_frames, axis=0) # 4D movie stub
        #colored_small_stack = depth_to_color(small_stack)
        colored_small_stack = depth_to_color(small_stack, total_num_slices = small_stack.shape[-1]) # attempt to bypass unresolved bug in depth_to_color
        time_lapse_movie = np.repeat(np.expand_dims(np.zeros_like(colored_small_stack), axis=0), num_binned_frames, axis=0) # 4D movie stub
    print(time_lapse_movie.shape)




    print('number of features is: ' + str(small_coordinate_array.shape[0]))
#    for feature_number in tqdm(range(small_coordinate_array.shape[0])):
    for feature_number in tqdm(range(10)):

        print('feature number ' + str(feature_number))
        #slice_my_feature = (slice(0,num_binned_frames),slice(0,5),slice(small_coordinate_array[feature_number,0],small_coordinate_array[feature_number,1]), slice(small_coordinate_array[feature_number,2], small_coordinate_array[feature_number,3])) 
        #print('feed into shape ' + str(time_lapse_movie[slice_my_feature].shape)) 
        #k = paste_neu_dynamics(feature_number)
        #time_lapse_movie[slice_my_feature] += k # Generating a tcxy movie
        #print('fed shape ' + str(k.shape))
        time_lapse_movie += elliptical_neu_dynamics(feature_number) # Generating a tcxy movie


    # prepare datasets to be optimal for FIJI in memory-constrained conditions



    full_fov = time_lapse_movie + np.expand_dims(colored_small_stack, axis=0)

    full_fov = np.divide(full_fov, np.nanmean(full_fov, axis=(0,2,3), keepdims=True)) # normalizing each color channel to stretch along the [0,1] range
    full_fov[:,3,:,:] = 0.75 * full_fov[:,3,:,:] # normalizing alpha value back to 0.75
    #full_fov = full_fov[:,slice(3),:,:]

    #label_me_movie(full_fov, frame_rate=sampling_rate/temporal_binning_factor[1], time_axis=0,color_axis=1,speedup_factor=1)

    # exchanging red and green channels such that superficial blood vessels will look red:
    red_channel = full_fov[:,0,:,:].copy() 
    full_fov[:,0,:,:] = full_fov[:,1,:,:] 
    full_fov[:,1,:,:] = red_channel


    #trace_saver(np.log(1+np.nanprod(1 + full_dff_trace, axis=0, keepdims=True)), np.nansum(full_spike_times, axis=0, keepdims=True), name_tag = 'sum_')

    #trace_saver(full_dff_trace, full_spike_times, name_tag = 'dff_')
    #png_saver(full_fov, time_axis=0, color_axis=1)
    
    movie_for_fiji(time_lapse_movie, colored_small_stack)

