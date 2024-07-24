import h5py
import zarr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
import seaborn as sns
import glob
from scipy.stats import zscore, pearsonr
from scipy.signal import correlate
from skimage.transform import downscale_local_mean

data_directory = 'D:' + os.sep + 'Downloads' + os.sep



neuronal_fn = data_directory + '2021_02_01_18_45_51_mouse_flipped_channels__1_FOV3_6100um_deep_512l_3x_mag_1850_sec_acq_with_FLIM_4Mcps_laser_pulses_with_TAG_0000[4-8]_neuronal_traces_tz_dff.zarr'
vascular_fn = data_directory + '2021_02_01_18_45_51_mouse_flipped_channels__1_FOV3_6100um_deep_512l_3x_mag_1850_sec_acq_with_FLIM_4Mcps_laser_pulses_with_TAG_0000[4-8]_vascular_1_Hz_lowpass_filtering_radon.zarr'
behvioural_fn = data_directory + 'mouse1_fov3-02012021184548-0001DLC_resnet_50_volumetricFeb22shuffle1_171500.h5'



drop_first_n_frames = 15 # drop the first 15 frames where baseline fluorescence is ill-defined

def stitch_datasets(file_wild_card, dataset_name, transpose_flag=False):
    fn_list = glob.glob(file_wild_card)
    num_files = len(fn_list)
    num_samples_per_file = np.zeros((num_files,1))
    sample_file_name = fn_list[0]    
    
    for fnum, fname in enumerate(fn_list):        
        z = zarr.open(fname, 'r')
        dynamics_array = np.array(z[dataset_name])
        if(transpose_flag):
            dynamics_array = np.transpose(dynamics_array)
        num_samples_per_file[fnum] = dynamics_array.shape[1]
        dynamics_array = dynamics_array[:,drop_first_n_frames:]
        if(fnum):
            output_array = np.concatenate((output_array, dynamics_array), axis=1)
        else:
            output_array = dynamics_array
    return output_array, num_samples_per_file


neu_dynamics, frame_number_list = stitch_datasets(neuronal_fn, 'neuronal_dff', transpose_flag=False)
#vas_dynamics, frame_number_list2 = stitch_datasets(vascular_fn, 'branching_vessel_photon_count', transpose_flag=True)
vas_dynamics, frame_number_list2 = stitch_datasets(vascular_fn, 'time_varying_vascular_diameter', transpose_flag=True)
mean_vas_dynamics = np.nanmean(vas_dynamics,axis=1,keepdims=True)
vas_dynamics = np.where(np.isnan(vas_dynamics), mean_vas_dynamics, vas_dynamics)




neu_dynamics = 100*neu_dynamics
print(f'frame_number_list is {frame_number_list}')
print(f'frame_number_list2 is {frame_number_list2}')

num_time_samples = neu_dynamics.shape[1]

sampling_rate = 30.02 # volumes per second

time_vector = np.linspace(0, num_time_samples-1, num=num_time_samples) * np.reciprocal(sampling_rate)

bv_df = pd.read_hdf(behvioural_fn)
num_behavioural_time_samples = bv_df.shape[0]

# cropping the neuro-vascular data to match the length of the behavioural data:
time_vector = time_vector[:2*num_behavioural_time_samples]
neu_dynamics = neu_dynamics[:,:2*num_behavioural_time_samples]
vas_dynamics = vas_dynamics[:,:2*num_behavioural_time_samples]

vas_dynamics_size_index = np.argsort(np.mean(vas_dynamics,axis=1))
sorted_vessels = vas_dynamics[vas_dynamics_size_index,:]

neu_dynamics_sort_index = np.argsort(np.sum(neu_dynamics[:,300:400],axis=1))
sorted_neurons = neu_dynamics[neu_dynamics_sort_index,:]


'''

zn = zarr.open(neuronal_fn, 'r')
zv = zarr.open(vascular_fn, 'r')

time_vector = np.array(zn['time_vector'])
neu_dynamics = 100 * np.array(zn['neuronal_dff'])
#vas_dynamics = np.transpose(np.array(zv['time_varying_vascular_diameter']))
vas_dynamics = np.transpose(np.array(zv['branching_vessel_photon_count']))
neu_dynamics = neu_dynamics[:,drop_first_n_frames:]
vas_dynamics = vas_dynamics[:,drop_first_n_frames:]

'''
'''
z_score_vas = zscore(vas_dynamics,axis=1)

mean_vas_dynamics = np.nanmean(vas_dynamics,axis=1,keepdims=True)
#test = np.nanmean(vas_dynamics,axis=1)
#print(f'shape of test vector is {test.shape}')
#normalized_vas_dynamics = np.divide(vas_dynamics,mean_vas_dynamics)
#normalized_vas_dynamics = vas_dynamics # skipping the normalization step!!!
normalized_vas_dynamics = 100 * np.divide( ( vas_dynamics - mean_vas_dynamics ) , mean_vas_dynamics)
normalized_vas_dynamics[normalized_vas_dynamics>50] = 50
normalized_vas_dynamics[normalized_vas_dynamics<-50] = -50

normalized_vas_dynamics[np.isnan(normalized_vas_dynamics)] = 0
'''
#time_vector = time_vector[:-drop_first_n_frames]




downsampled_time_vector = time_vector[0:-1:2]
ticked_time_vector = np.floor(time_vector[0:-1:100])
print(f'shape of ticked_time_vector is {ticked_time_vector.shape}')
cropped_df = bv_df.iloc[:len(downsampled_time_vector)]

print(time_vector)
print('fsjlksjlk')
print(ticked_time_vector)

print(time_vector.shape)
print(neu_dynamics.shape)
print(vas_dynamics.shape)
print(downsampled_time_vector.shape)


#print(cropped_df)
#print(cropped_df.columns.values)

'''
paws_df = pd.DataFrame()
paws_df['left_x'] = cropped_df['DLC_resnet_50_volumetricFeb22shuffle1_171500']['left_paw']['x']
paws_df['left_y'] = cropped_df['DLC_resnet_50_volumetricFeb22shuffle1_171500']['left_paw']['y']
paws_df['right_x'] = cropped_df['DLC_resnet_50_volumetricFeb22shuffle1_171500']['right_paw']['x']
paws_df['right_y'] = cropped_df['DLC_resnet_50_volumetricFeb22shuffle1_171500']['right_paw']['y']
paws_df['snout_x'] = cropped_df['DLC_resnet_50_volumetricFeb22shuffle1_171500']['snout']['x']
paws_df['snout_y'] = cropped_df['DLC_resnet_50_volumetricFeb22shuffle1_171500']['snout']['y']

y_values = paws_df.columns.values
#y_values = paws_df.index.values

no_time_paws = paws_df.copy()
#z_paws = zscore(no_time_paws, axis=1)
'''

motion_array = cropped_df['DLC_resnet_50_volumetricFeb22shuffle1_171500'].to_numpy()

normalized_motion = np.divide(motion_array, np.max(np.abs(motion_array), axis=1, keepdims=True))

motion_score = np.sum(np.square(normalized_motion), axis=1)
motion_score = motion_score-motion_score.min()
motion_score = motion_score / motion_score.max()
motion_score = np.expand_dims(motion_score, axis=1)

ds_vas = downscale_local_mean(sorted_vessels, (1,2))
ds_neu = downscale_local_mean(sorted_neurons, (1,2))


print(ds_neu.shape)
print(ds_vas.shape)
print(ds_neu.dtype)
print(motion_score.dtype)
vas_correlation = correlate(ds_vas, motion_score, mode='same')
neu_correlation = correlate(ds_neu, motion_score, mode='same')
print(f'vas_correlation is {vas_correlation}')

#print(neu_pearsons.shape)
neu_vas_correlation = correlate(sorted_vessels.T, sorted_neurons.T, mode='valid')
plt.imshow(vas_correlation)
plt.show()
plt.imshow(neu_correlation)
plt.show()
plt.imshow(neu_vas_correlation)
plt.show()


'''
z_paws = zscore(cropped_df,axis=1)


paws_df['time [s]'] = downsampled_time_vector
'''


#wide_paws = paws_df.pivot(columns=y_values)
#wide_paws = paws_df.pivot(index='time [s]',columns=y_values)

#sns.lineplot(data=paws_df)
#sns.lineplot(data=wide_paws)
'''
sns.lineplot(data=paws_df, x="time [s]", y="snout_x")
sns.lineplot(data=paws_df, x="time [s]", y="left_x")
plt.show()
'''
#print(bv_df)

### generate neuro-vascular dataframes

neu_df = pd.DataFrame(data=sorted_neurons.T) 
neu_df['time [s]'] = np.floor(time_vector)
print(neu_df)

vas_df = pd.DataFrame(data=sorted_vessels)

#wide_neu_df = neu_df.pivot(columns="time [s]")

########## plot stuff

fig, axes = plt.subplots(3, 1, figsize=(10, 10))
fig.suptitle('Neuro-vascular traces with behavioural context')

#neu_pos = axes[0].imshow(neu_dynamics, cmap='cividis')
neu_cbar_dict = {
  "label": "\u0394F/F [%]"
}
vas_cbar_dict = {
  "label": r'diameter [ $\mu m$]'
}

sns.heatmap(ax=axes[0],data=neu_df.T, cmap='cividis', cbar=True, cbar_kws=neu_cbar_dict, xticklabels=False)
sns.heatmap(ax=axes[1],data=vas_df, cmap='plasma', cbar=True, cbar_kws=vas_cbar_dict, xticklabels=False)
axes[2].plot(downsampled_time_vector, motion_score)
plt.xlabel('Time [s]')
plt.ylabel('Motion score')

#sns.heatmap(ax=axes[0],data=neu_df.T, xticks="time [s]", cmap='cividis', cbar=True)


#vas_pos = axes[1].imshow(normalized_vas_dynamics, cmap='inferno',aspect='auto')

#vas_pos = axes[1].imshow(z_score_vas, cmap='inferno',aspect='auto')
#vas_pos = axes[1].imshow(vas_dynamics, cmap='inferno',aspect='auto')


#sns.lineplot(ax=axes[2],data=paws_df, x="time [s]", y="right_x")
#sns.lineplot(ax=axes[2],data=cropped_df)
#sns.lineplot(ax=axes[2],data=paws_df, x="time [s]", y="right_x")

'''
for ax_number in range(2):
    #axes[ax_number].set_xticks(np.arange(len(ticked_time_vector)))
    axes[ax_number].set_xticklabels(ticked_time_vector)
'''

#fig.colorbar(neu_pos, ax=axes[0], label="\u0394F/F [%]")
#fig.colorbar(vas_pos, ax=axes[1], label="diameter [ \mu m]")


plt.show()

fig2 = plt.plot(time_vector, np.sum(sorted_neurons,axis=0))

plt.show()

fig3 = plt.plot(time_vector, np.sum(sorted_vessels,axis=0))

plt.show()
