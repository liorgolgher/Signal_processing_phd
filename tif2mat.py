import tifffile as tf
from scipy import io as sio
import numpy as np

fld = 'D:/Downloads/'
#fn = fld + 'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_1024l_1slow_3p5x_zoom_+100um_heigh_summed_bc_t_FOV1_062_summed_bc_moments_binarized_top180_slices_rs512x2048_3x_dilated.tiff'
#fn = fld + 'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_1024l_1slow_3p5x_zoom_+100um_heigh_summed_bc_t_FOV1_062_summed_bc_top180_slices_rs512x2048_for_vg_gamma2_rs_512x512x180_clij2_percentile_binary_3d_median_2x2x2.tif'
#fn = fld + 'binary_mask_100x130x180.tif'
#fn = fld + 'ref_2020_01_13_512x512x150_binary_otsu.tif'
fn = fld + '2021_02_01_19_19_39_mouse_flipped_channels__1_FOV3_6100um_deep_512l_3x_mag_1850_sec_acq_with_FLIM_4Mcps_laser_pulses_with_TAG_00053_summed_bc_improved_vascular_contrast_equalized.tif'


#save_fld = 'D:/code/VascularGraph/'
#sfn = save_fld + 'binary_mask.mat'
sfn = fn.replace('.tif','.mat')

segmentation_mask = np.array(tf.imread(fn))


segmentation_mask = np.moveaxis(segmentation_mask, 0, -1) # FiJI turns the depth dimension into the first (0th) dimension. Here we fix it back


sio.savemat(sfn, {'seg': segmentation_mask})

print('all done')