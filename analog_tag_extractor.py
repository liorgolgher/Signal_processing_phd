import tifffile as tf
from scipy import io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, find_peaks_cwt
import zarr
from numcodecs import Blosc, Zstd
from skimage.util import crop
import time



def save_dataset(save_file_name,group_name,dataset):
    root = zarr.open_group(save_file_name, mode='w')
    fill_me = root.require_group(group_name)
    root[group_name] = dataset

zarr.storage.default_compressor = Zstd(level=3)
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)



def calc_peaks_in_single_line(single_line, min_peak_distance):
    ats = -single_line
    #ats = ats - ats.min()
    ats = np.divide(ats, ats.max())

    peak_indices, _ = find_peaks(ats, height=0.5, distance=min_peak_distance, width=(0,5)) # rejecting false peaks by their distance and prominence

    output_stream = np.zeros_like(ats)
    output_stream[peak_indices] = 1
    
    return output_stream

def calc_peaks_in_single_frame(single_frame, min_peak_distance):
    ats = np.ndarray.flatten(-single_frame)
    #ats = ats - ats.min()
    ats = np.divide(ats, ats.max())
    threshold_value = 0.8

    percentile_threshold = np.percentile(ats, 97)

    peak_indices, _ = find_peaks(ats, height=0.5, distance=min_peak_distance, width=(0,5)) # rejecting false peaks by their distance and prominence
    '''
    plt.plot(ats)
    plt.plot(peak_indices, ats[peak_indices], "x")
    plt.show()
    '''
    #plt.hist(np.diff(peak_indices),bins=100)
    #plt.show()
    output_stream = np.zeros_like(ats)
    output_stream[peak_indices] = 1
    
    return np.reshape(output_stream, single_frame.shape)

def calc_peaks_in_movie(movie, min_peak_distance):
    ats = np.ndarray.flatten(-movie)
    #ats = ats - ats.min()
    ats = np.divide(ats, ats.max())
    threshold_value = 0.8


    percentile_threshold = np.percentile(ats, 97)

    peak_indices, _ = find_peaks(ats, height=0.5, distance=min_peak_distance, width=(0,5)) # rejecting false peaks by their distance and prominence
    '''
    plt.plot(ats)
    plt.plot(peak_indices, ats[peak_indices], "x")
    plt.show()
    '''
    #plt.hist(np.diff(peak_indices),bins=100)
    #plt.show()
    output_stream = np.zeros_like(ats)
    output_stream[peak_indices] = 1
    
    return np.reshape(output_stream, movie.shape)

def calc_diff_histogram(input_array):
    #dummy_histogram, global_bin_edges = np.histogram(np.diff(input_array, axis=1), bins=100)
    #global_histogram = np.zeros_like(dummy_histogram)
    global_bin_edges = np.arange(200)
    global_histogram = np.zeros((199,),dtype='uint64')
    n = 0
    for frame_number in range(input_array.shape[2]):
        for line_number in range(input_array.shape[0]):
            peaks_in_this_lines = np.nonzero(np.squeeze(input_array[line_number, :, frame_number]))
            distance_between_peaks = np.diff(peaks_in_this_lines)
            local_histogram, _ = np.histogram(distance_between_peaks, bins=global_bin_edges)
            global_histogram += np.uint64(local_histogram)
            if(n<0):
                print(peaks_in_this_lines)
                print(distance_between_peaks)
                print(local_histogram)
                print(global_histogram)
                n += 1
    return global_histogram, global_bin_edges


if __name__=='__main__':

    fld = os.path.join('D:' + os.sep, 'Downloads')
    #fn = os.path.join(fld ,'Analog_TAG_P1P2T_GCaMP7_mouse_2x_zoom_1024l_no_avg_00001.tif') # 1217 frames
    #fn = os.path.join(fld ,'Analog_TAG_P1P2T_GCaMP7_mouse_2x_zoom_512l1024c_no_avg_00001.tif') # 2076 frames
    #fn = os.path.join(fld ,'mouse2_FITC_GCaMP6s_and_Texas_Red_P1P2_analog_TAG_6x_zoom_FOV1_43p9ns_dwell_time_00004.tif') # time series looks corrupt
    #fn = os.path.join(fld ,'P1_P2_analogTAG_fixed_sample_189k_62p_1x_zoom_43p9ns_dwell_time_-300um_deep_FOV3_WITH_3_SAVED_CHANNELS_00002.tif')# 1800 frames
    #fn = os.path.join(fld ,'P1_P2_analogTAG_fixed_sample_189k_62p_1x_zoom_43p9ns_dwell_time_-300um_deep_FOV3_WITH_3_SAVED_CHANNELS_00003.tif')# 210 frames
    #fn = os.path.join(fld ,'P1_P2_analogTAG_fixed_sample_189k_62p_1x_zoom_43p9ns_dwell_time_-300um_deep_FOV3_WITH_3_SAVED_CHANNELS_00005.tif')# 1500 frames
    fn = os.path.join(fld ,'P1_P2_analogTAG_fixed_sample_189k_62p_1x_zoom_43p9ns_dwell_time_-300um_deep_FOV3_WITH_3_SAVED_CHANNELS_00004.tif')# 330 frames

    #save_fn = fn[:-4] + '_tag_indices.zarr'
    save_fn = fn.replace('.tif' ,'_tag_indices.npz')

    tag_image = tf.imread(fn, key=range(2,330,3))  # Every third frame is a TAG sync signal frame

    num_frames = tag_image.shape[0]

    print(f'tag_image shape is {tag_image.shape}')

    #tag_image = np.moveaxis(tag_image,0,2)
    tag_image = np.moveaxis(tag_image,1,0)

    print(f'tag_image shape is {tag_image.shape}')


    #raw_tag_signal = tag_image.ravel()

    nominal_distance_between_peaks = 150
    minimal_distance_between_peaks = int(np.floor(nominal_distance_between_peaks*0.8))
    print(f'min peak distance is {minimal_distance_between_peaks}')


    '''
    right_pad_array = np.ones((tag_image.shape[0],minimal_distance_between_peaks,tag_image.shape[2]))

    right_pad_array = np.einsum('ijk,ik->ijk',right_pad_array,np.squeeze(tag_image[:,-1,:]))

    padded_tag_image = np.concatenate((tag_image,right_pad_array),axis=1)
    '''

    pad_width =  ((minimal_distance_between_peaks, minimal_distance_between_peaks), (0, 0), (0, 0)) 
    print(pad_width)

    padded_tag_image = np.pad(tag_image, pad_width=pad_width, mode='edge')
    padded_peak_pic = np.zeros(padded_tag_image.shape, dtype='bool')

    #padded_tag_image = np.moveaxis(padded_tag_image,1,0)

    '''
    frame_num = 4

    plt.imshow(padded_tag_image[:,frame_num,:])
    plt.show()
    '''

    loop_ppp = padded_peak_pic.copy()
    double_loop_ppp = padded_peak_pic.copy()

    vectorized_start_time = time.time()
    padded_peak_pic = calc_peaks_in_movie(padded_tag_image, minimal_distance_between_peaks)
    cropped_peak_pic = crop(padded_peak_pic, crop_width=pad_width)
    print(f'vectorized running time is {(time.time() - vectorized_start_time)}' )

    loop_start_time = time.time()
    for frame_num in range(num_frames):
        loop_ppp[:,frame_num,:] = calc_peaks_in_single_frame(padded_tag_image[:,frame_num,:], minimal_distance_between_peaks)
    cropped_lppp = crop(loop_ppp, crop_width=pad_width)
    print(f'loop running time is {(time.time() - loop_start_time)}' )



    double_loop_start_time = time.time()
    for frame_num in range(num_frames):
        for line_num in range(loop_ppp.shape[0]):
            double_loop_ppp[line_num,frame_num,:] = calc_peaks_in_single_line(padded_tag_image[line_num,frame_num,:], minimal_distance_between_peaks)
    cropped_dlppp = crop(double_loop_ppp, crop_width=pad_width)
    print(f'double loop running time is {(time.time() - double_loop_start_time)}' )

    cropped_peak_pic    = np.moveaxis(cropped_peak_pic, 1,2)
    cropped_lppp        = np.moveaxis(cropped_lppp,     1,2)
    cropped_dlppp       = np.moveaxis(cropped_dlppp,    1,2)
    
    vectorized_histogram, vectorized_bin_edges = calc_diff_histogram(cropped_peak_pic)
    looped_histogram, looped_bin_edges = calc_diff_histogram(cropped_lppp)
    

    loop_histogram_start_time = time.time()
    double_looped_histogram, double_looped_bin_edges = calc_diff_histogram(cropped_dlppp)
    print(f'loop histogram running time is {(time.time() - loop_histogram_start_time)}' )
    



    plt.plot(vectorized_bin_edges[:-1],vectorized_histogram)
    plt.plot(looped_bin_edges[:-1],looped_histogram)
    plt.plot(double_looped_bin_edges[:-1],double_looped_histogram)
    plt.legend(('vectorized','looped','double_looped'))
    plt.show()




    plt.imshow(cropped_dlppp[:,:,frame_num])
    plt.show()

        
    '''
    #ats = np.abs(padded_tag_image.ravel())
    ats = -padded_tag_image.ravel()

    #crude_tag_times, _ = find_peaks(raw_tag_signal, prominence=40)
    #time_diff_histogram, bin_edges, _ = plt.hist(np.diff(crude_tag_times), bins=100)
    #plt.show()
    #this step shows that the nominal distance between adjacent peaks is roughly 120 pixels ~ 5290 ns with a dwell time of 44ns per pixel

    #fine_tag_indices, _ = find_peaks(ats, prominence=15, distance=70, width=(0,5)) # rejecting false peaks by their distance and prominence
    fine_tag_indices, _ = find_peaks(ats, prominence=15, distance=70) # rejecting false peaks by their distance and prominence

    '''
    '''
    plt.plot(ats)
    plt.plot(fine_tag_indices, ats[fine_tag_indices], "x")
    plt.show()
    '''

    with open(save_fn, 'w') as f:
        np.savez_compressed(save_fn, looped_movie=cropped_lppp, double_looped_movie=cropped_dlppp, bin_edges=double_looped_bin_edges, peak_distance_histogram=double_looped_histogram)
    print(cropped_dlppp.shape)

    print('all done')