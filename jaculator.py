# checking how well do we fair against Na Ji's Bessel beam neurovascular imaging

import h5py
import zarr
import numpy as np
#from scipy import signal

import os
#import matplotlib.pyplot as plt
import time
import glob
#from scipy.ndimage.filters import gaussian_filter
from skimage.transform import downscale_local_mean, rescale
from skimage.filters import gaussian
from numcodecs import Zstd
import tifffile as tf
from dask import array as da
from dask import config as da_config
import multiprocessing





def norm_stack_intensity(big_stack):
# assume that the returned stack should be summed along the first (file name) dimension and normalized along the last (axial) dimension
   axial_intensity_profile = np.sum(big_stack,axis=(0,1),dtype='float64')
   reciprocal_axial_intensity_profile = np.reciprocal(axial_intensity_profile)
   reciprocal_axial_intensity_profile = np.divide(reciprocal_axial_intensity_profile , np.max(reciprocal_axial_intensity_profile))
   return reciprocal_axial_intensity_profile


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

def h5_get_stacks(file_wc):
    fn_list = glob.glob(file_wc)
    num_data_files = len(fn_list)

    print(fn_list[0])


    with h5py.File(fn_list[0],mode='r') as f:
        num_lines, num_columns, num_slices = f['Summed Stack']['Channel 1'].shape
    all_stacks = np.zeros((num_data_files,num_lines,num_columns,num_slices), dtype='int16')
    for fnum, fname in enumerate(fn_list):
        with h5py.File(fname,mode='r') as f:
            all_stacks[fnum,:] = np.array(f['Summed Stack']['Channel 1'])


    #sfn = fname[:-5] + '_summed_bc.hdf5'


def get_stacks(file_wc, preset_num_slices=512):
    fn_list = glob.glob(file_wc)
    num_data_files = len(fn_list)
    print(str(num_data_files) + ' files found')

    if num_data_files > 100: # if we're summing many input data files, leave room for many photons
        all_stacks_dtype = 'int64'
    else:
        all_stacks_dtype = 'int16'

    z = zarr.open(fn_list[0],mode='r')
    num_lines, num_columns, num_slices = z['Summed Stack']['Channel 1'][:,:,:preset_num_slices].shape
    all_stacks = np.zeros((num_lines,num_columns,num_slices), dtype=all_stacks_dtype)
    for fnum, fname in enumerate(fn_list):
        print(fname)
        z = zarr.open(fname,mode='r')
        all_stacks = all_stacks + np.array(z['Summed Stack']['Channel 1'][:,:,:num_slices])


    sfn = fname.replace('.zarr','_jaculated_1000_two_sides.zarr')


    return all_stacks, sfn

if __name__=='__main__':

    #da_config.set(pool=multiprocessing.pool.ThreadPool(8))

    num_cropped_frames = 1800
    time_step_size = 25
    y_step_size = 256

    zarr.storage.default_compressor = Zstd(level=3)


    fld_name = 'D:/Downloads/'

    file_wild_card = fld_name + 'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_512l_1slow_3p5x_zoom_+100um_height_FOV1_034.zarr'

    reference_volume_fn = fld_name + 'ref_2020_01_13.tif'
    reference_volume = np.array(tf.imread(reference_volume_fn))
    reference_volume = np.moveaxis(reference_volume, 0, -1) # FiJI turns the depth dimension into the first (0th) dimension. Here we fix it back

    summed_stack, save_fn = get_stacks(file_wild_card, preset_num_slices=reference_volume.shape[2])
    reciprocal_axial_profile = norm_stack_intensity(summed_stack)

    fn_list = glob.glob(file_wild_card)
    z = zarr.open(fn_list[0],mode='r')

    holy_moly = np.zeros((num_cropped_frames,reference_volume.shape[0],reference_volume.shape[1]))
    side_dish = np.zeros((num_cropped_frames,reference_volume.shape[0],reference_volume.shape[2]))
    
    print(f'side_dish shape: {side_dish.shape}')

    num_iterations = int(np.floor(num_cropped_frames/time_step_size))
    num_y_iterations = int(np.floor(2048/y_step_size))

    print(f'num iterations: {num_iterations}')
    print(f'num_y_iterations: {num_y_iterations}')

    for time_iter_num in range(num_iterations):
        for y_iter_num in range(num_y_iterations):
            print(f'iteration {time_iter_num}')
            #dz = da.from_zarr(z['Full Stack']['Channel 1'][slice(time_iter_num*time_step_size,(time_iter_num+1)*time_step_size),:,:,slice(0,180)])
            #vaseline = gaussian(dz, sigma=(2,6,2,3))
            vaseline = gaussian(z['Full Stack']['Channel 1'][slice(time_iter_num*time_step_size,(time_iter_num+1)*time_step_size),:,slice(y_iter_num*y_step_size, (y_iter_num+1)*y_step_size),slice(0,180)], sigma=(2,3,12,15))

            holy_moly[slice(time_iter_num*time_step_size, (time_iter_num+1)*time_step_size),:,slice(y_iter_num*y_step_size, (y_iter_num+1)*y_step_size)] = np.einsum('ijkl,jkl,l->ijk',vaseline, reference_volume[:,slice(y_iter_num*y_step_size, (y_iter_num+1)*y_step_size),:], reciprocal_axial_profile)
            side_dish[slice(time_iter_num*time_step_size, (time_iter_num+1)*time_step_size),:] += np.einsum('ijkl,jkl,l->ikl',vaseline, reference_volume[:,slice(y_iter_num*y_step_size, (y_iter_num+1)*y_step_size),:], reciprocal_axial_profile)
            #holy_moly[slice(time_iter_num*time_step_size, (time_iter_num+1)*time_step_size),:,slice(y_iter_num*y_step_size, (y_iter_num+1)*y_step_size)] = np.einsum('ijkl,l->ijk', vaseline, reciprocal_axial_profile)
            #side_dish[slice(time_iter_num*time_step_size, (time_iter_num+1)*time_step_size),:] += np.einsum('ijkl,l->ijl',vaseline, reciprocal_axial_profile)


    h5_save_fn = save_fn.replace('.zarr','_'+str(num_cropped_frames)+'_frames_outcome.hdf5')

    h5_save_dataset(h5_save_fn, '180_slices',holy_moly)
    h5_save_dataset(h5_save_fn, '180_slices_side_view',side_dish)

    save_dataset(save_fn,'180_slices_'+str(num_cropped_frames)+'_frames', holy_moly)
    save_dataset(save_fn, '180_slices_side_view_'+str(num_cropped_frames)+'_frames',side_dish)


    '''
    save_dataset(save_fn,'summed_bc_stack', normed_summed_stack)
    save_dataset(save_fn,'reciprocal_axial_intensity_profile', recip_axial_profile)
    save_dataset(save_fn,'summed_stack', summed_stack)


    #save_dataset(save_fn,'summed_bc',sss) # to be used once the zarr saving thingy is configured properly!
    h5_save_fn = save_fn[:-5] + '_summed_bc.hdf5' # temporary workaround until zarr is on
    h5_save_fn = save_fn.replace('.zarr','_summed_bc.hdf5')
    h5_save_dataset(h5_save_fn,'summed_bc',normed_summed_stack)
    h5_save_dataset(h5_save_fn,'summed_stack',summed_stack.astype('int16'))
    h5_save_dataset(h5_save_fn,'recip_axial_profile',recip_axial_profile)
    '''