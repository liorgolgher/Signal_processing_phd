import h5py
import numpy as np
import os

fld = 'D:\Downloads'

fname = os.path.join(fld , 'TLP1_189k62p_GCaMP7_mouse_FOV1_1x_zoom_2048l_0p75slow_calibrated_0u_summed_bc_m_depth753_summed_bc.hdf5')

#h5_save_fn = fname.replace('.hdf5','_split.hdf5')
h5_save_fn = fname.replace('.hdf5','_single_plane.hdf5')

def h5_save_dataset(save_file_name,dataset_name,dataset):
    with h5py.File(save_file_name, 'a') as fout:
        fout.require_dataset(dtype=dataset.dtype,
                             compression="gzip",
                             chunks=True,
                             name=dataset_name,
                             shape=dataset.shape)
        fout[dataset_name][...] = dataset


with h5py.File(fname,mode='r') as f:
    #sbc_top = np.array(f['summed_bc'][:,:,:256])
    #sbc_bottom = np.array(f['summed_bc'][:,:,256:])
    #summed_plane = np.sum(np.array(f['summed_stack']),axis=2).astype("float64")
    ss_top = np.array(f['summed_stack'][:,:,:256]).astype("float64")


#h5_save_dataset(h5_save_fn,'top_bc_slices',sbc_top)
#h5_save_dataset(h5_save_fn,'bottom_bc_slices',sbc_bottom)
#h5_save_dataset(h5_save_fn,'summed_plane',summed_plane)
h5_save_dataset(h5_save_fn,'summed_stack',ss_top)