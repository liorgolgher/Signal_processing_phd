import h5py
import zarr
import numpy as np
#from scipy import signal

import os
#import matplotlib.pyplot as plt
import time
import glob
#from scipy.ndimage.filters import gaussian_filter
#from skimage.transform import downscale_local_mean, rescale
from numcodecs import Zstd





zarr.storage.default_compressor = Zstd(level=3)


#fld_name = '/data/Lior/lst/2020/2020_02_03/FOV1/power8_cooked/'
#fld_name = '/data/Lior/lst/2020/2020_01_13/'
#fld_name = '/data/Lior/lst/2020/2020_03_04/seg4/'
fld_name = 'D:/Downloads' + os.sep + '2020_11_23/'

#file_wild_card = fld_name + 'TLP2_189k62p_Tie1GCaMP6s_mouse_3p5x_zoom_512l_1slow_FOV2_new_TAG_OFF_some_depth_2??.zarr'
#file_wild_card = fld_name + 'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_512l_1slow_3p5x_zoom_+200um_height_FOV1_???.zarr'
#file_wild_card = fld_name + 'TLP1_189k62p_GCaMP7_mouse_FOV1_1x_zoom_512l_0p75slow_calibrated_0um_depth???.zarr'
#file_wild_card = fld_name + 'TLP2_189k62p_Tie1GCaMP6s_mouse_3p5x_zoom_512l_1slow_FOV2_new_40um_deep_???.zarr'
#file_wild_card = fld_name + 'TLP2_187k62p_pollen_1x_zoom_512l_1slow_???.zarr'
#file_wild_card = fld_name + 'TLP2_189k62p_1250nm_Alexa680_mouse_2048l_1slow_1p75x_zoom_FOV1_150um_higher_013*.zarr'
#file_wild_card = fld_name + 'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_512l_1slow_3p5x_zoom_+200um_height_FOV1_TAG_OFF_SYNC_ON_+20um_deep_comparable_to_200um_???.zarr'

#file_wild_card = fld_name + 'TLP2_189k62p_Texas_Red_GCaMP7_mouse_3p5zoom_1slow_512l_100_short_sweeps_setup01*.zarr'
#file_wild_card = fld_name + 'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_512l_1slow_3p5x_zoom_+100um_height_FOV1_034.zarr'
#file_wild_card = fld_name + '256lines_dump_2_seconds_21_24_07_single_edge_per_channel_uncompressed.zarr'
#file_wild_card = fld_name + '1024lines_dump_15_seconds_21_11_14_2edges_DEBUG.zarr'
#file_wild_card = fld_name + '1024lines_dump_15_seconds_21_11_14_2edges_DEBUG.zarr'
#file_wild_card = fld_name + '512l_1x_mag_5_sec_acq_pollen_no_FLIM_with_TAG_lens_16_11_26.zarr'
file_wild_card = fld_name + '512l_1x_mag_45_sec_acq_pollen_no_FLIM_16_04_02_1slice.zarr'
#file_wild_card = fld_name + '512l_1x_mag_45_sec_acq_pollen_no_FLIM_16_04_02_DEBUG.zarr'
#file_wild_card = fld_name + 'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_256l_0p5slow_3p5x_zoom_+100um_height_max_power_FOV1_*.zarr'
#file_wild_card = fld_name + '*2x*.zarr'


def norm_stack_intensity(big_stack):
# assume that the returned stack should be summed along the first (file name) dimension and normalized along the last (axial) dimension
   axial_intensity_profile = np.sum(big_stack-big_stack.min(),axis=(0,1),dtype='float64')
   reciprocal_axial_intensity_profile = np.reciprocal(axial_intensity_profile)
   reciprocal_axial_intensity_profile = np.divide(reciprocal_axial_intensity_profile , np.max(reciprocal_axial_intensity_profile))
   return np.einsum('ijk,k->ijk', big_stack.astype('float64'), reciprocal_axial_intensity_profile), reciprocal_axial_intensity_profile


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


def get_stacks(file_wc, channel_number = int(1)):
    fn_list = glob.glob(file_wc)
    num_data_files = len(fn_list)
    print(str(num_data_files) + ' files found')

    if num_data_files > 100: # if we're summing many input data files, leave room for many photons
        all_stacks_dtype = 'int64'
    else:
        all_stacks_dtype = 'int16'




    #print(fn_list[0])
    #z = zarr.open(str(file))
    #a = np.array(z['Summed Stack']['Channel 1'], dtype='float64')
    #b = np.sum(a,axis=(0,1), dtype='float64')
    #c = np.einsum('ijk,k->ijk', a, np.reciprocal(b))

    z = zarr.open(fn_list[0],mode='r')
    channel_name = 'Channel ' + str(channel_number)
    try:
        num_lines, num_columns, num_slices = z['Summed Stack'][channel_name].shape
        all_stacks = np.zeros((num_lines,num_columns,num_slices), dtype=all_stacks_dtype)
        planar_flag = False
    except ValueError:
        num_lines, num_columns = z['Summed Stack'][channel_name].shape
        all_stacks = np.zeros((num_lines,num_columns), dtype=all_stacks_dtype)
        planar_flag = True
    
    for fnum, fname in enumerate(fn_list):
        print(fname)
        z = zarr.open(fname,mode='r')
        print(z.keys())
        print(z['Summed Stack'].keys())
        
        all_stacks = all_stacks + np.array(z['Summed Stack'][channel_name])


    #sfn = fname[:-15] +     '_summed_bc_' + fname[-15:-5] + '.zarr'
    sfn = fname.replace('.zarr','_summed_bc.zarr')


    return all_stacks, sfn, planar_flag


summed_stack, save_fn, planar_flag = get_stacks(file_wild_card)

if planar_flag:
    normed_summed_stack = summed_stack
    recip_axial_profile = np.ones((1,))
else:
    normed_summed_stack, recip_axial_profile = norm_stack_intensity(summed_stack)


save_dataset(save_fn,'summed_bc_stack', normed_summed_stack)
save_dataset(save_fn,'reciprocal_axial_intensity_profile', recip_axial_profile)
save_dataset(save_fn,'summed_stack', summed_stack)

    

#save_dataset(save_fn,'summed_bc',sss) # to be used once the zarr saving thingy is configured properly!
h5_save_fn = save_fn[:-5] + '_summed_bc.hdf5' # temporary workaround until zarr is on
h5_save_fn = save_fn.replace('.zarr','_summed_bc.hdf5')
h5_save_dataset(h5_save_fn,'summed_bc',normed_summed_stack)
h5_save_dataset(h5_save_fn,'summed_stack',summed_stack.astype('int16'))
h5_save_dataset(h5_save_fn,'recip_axial_profile',recip_axial_profile)

num_channels = int(2)

if num_channels>1:
    summed_stack2, save_fn, planar_flag = get_stacks(file_wild_card, channel_number=2)
    save_dataset(save_fn,'summed_stack_CH2', summed_stack2)
    h5_save_dataset(h5_save_fn,'summed_stack_CH2',summed_stack2.astype('int16'))


'''
rsss = rescale(sss,(4,1,1))
rsss = rescale(sss,(8,1,1))
save_dataset(h5_save_fn,'summed_bc_rs1024x1024x512',rsss)
'''

'''
grs = gaussian_filter(rsss,2,2,2))
save_dataset(h5_save_fn,'summed_bc_rs1024x1024x512_gauss3x3x3',grs)
'''