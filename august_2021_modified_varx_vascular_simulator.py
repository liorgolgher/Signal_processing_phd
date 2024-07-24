# The relevant environment is varx

# Generate a simple vascular relationship network, then try to infer it back using vector autoregression

'''
import tifffile as tf
from scipy import io as sio
from scipy import signal
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, find_peaks_cwt
import zarr
from numcodecs import Blosc, Zstd 
'''
	
# VAR example
from statsmodels.tsa.vector_ar.var_model import VAR
#from random import random
from numpy.random import default_rng
#import zarr
import numpy as np
#from numcodecs import Blosc, Zstd 
from scipy import stats, signal
import matplotlib.pyplot as plt
import pickle
import os

rng = default_rng()

num_time_samples = 60000 # 30000 # 4000 # 3000#
num_vessels = 8 # start with a oversimplified network composed of 8 vessel segments and two bifurcations as depicted above. Blood flow in vessels 0,1,5 is measurable and is driven by blood flow in vessels 2,6 
sampling_frequency_hz = float(30) # 30 volumes per second smoothed to 10 samples per second
rbc_velocity = 4.5 # start with a constant velocity of red blood cells across the entire network. RBC transit time and inter-vascular latency are dictated by vascular distance d (t=d/v_rbc), with no dispersion.
heartbeat_frequency_hz = float(10)*0.01 # should be 10Hz ± 5Hz. Here changed to artificially lower values to satisfy sufficient sampling

num_training_samples = 10000 # 1500 # 1500 # 5000
num_estimated_time_steps = 1500 # num_estimated_time_steps must be smaller than num_time_samples

#step_sample_time = 2500 # model a step function that starts at this sample

'''
# contrived dataset with dependency
data = list()

for j in range(num_time_samples):
    v1 = np.divide(j , num_time_samples) + 1e-1*random() + np.sin(2*np.pi*sampling_frequency_hz*j)
    v2 = 0.5*v1 + 1e-1*random()
    row = [v1, v2]
    data.append(row)
'''

#time_vector = np.linspace(0, num_time_samples*sampling_frequency_hz,num_time_samples)
'''
v1 = np.sin(2*np.pi*sampling_frequency_hz*time_vector*0.03) + 2
v1 += 0.2* (1 + np.random.randn(*v1.shape))
print(v1)
print(stats.describe(v1))
lag_value = 20 * sampling_frequency_hz
v2 = 0.5*np.roll(v1,lag_value) + 1e-0*random()
cbf = np.transpose( np.stack((v1,v2))   )
plt.plot(time_vector,cbf)
plt.show()
used_cbf = cbf[lag_value:-lag_value,:]
forecast_time_vector = time_vector[-lag_value] + time_vector
short_time_vector = time_vector[lag_value:-lag_value]
print(cbf.shape)
'''


'''

def save_dataset(save_file_name,group_name,dataset):
    root = zarr.open_group(save_file_name, mode='w')
    fill_me = root.require_group(group_name)
    root[group_name] = dataset

zarr.storage.default_compressor = Zstd(level=3)
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

'''

#vascular_lengths = np.abs(rng.normal(30, 10, size=(num_vessels,)))
vascular_lengths = np.abs(rng.normal(80, 30, size=(num_vessels,)))
#vascular_lengths = np.abs(rng.normal(10, 30, size=(num_vessels,)))


#vascular_lengths = np.abs(np.random.normal(5, 1, size=(num_vessels,)))

print(f'vascular_lengths are {vascular_lengths}')

#vascular_latency_vector = np.rint((vascular_lengths*sampling_frequency_hz / rbc_velocity), dtype='int64')
vascular_latency_vector = np.rint(vascular_lengths * sampling_frequency_hz * np.reciprocal(rbc_velocity)).astype('int64')
print(f'vascular_latency_vector is {vascular_latency_vector}')

vascular_latency_matrix = np.zeros((num_vessels,num_vessels), dtype='int64')
vascular_coupling_matrix = np.zeros((num_vessels,num_vessels))


### Build vascular relationship network
#       ___2___
#     1/       \ 3
#__0__/         \____4___
#     \         / 
#     5\___6___/ 7
#

vessel_names = ('Output vein' , 'Venule (a)', 'Capillary (a)' , 'Arteriole (a)', 'Artery', 'Venule (b)', 'Capillary (b)', 'Arteriole (b)')

max_time_in_seconds = np.divide(num_time_samples,sampling_frequency_hz)
print(f'max time is {max_time_in_seconds}')
print(f'step size is {np.reciprocal(sampling_frequency_hz)}')

time_vector = np.arange(0, max_time_in_seconds, np.reciprocal(sampling_frequency_hz) )
pre_filtered_time_vector = np.arange(-max_time_in_seconds*0.5, max_time_in_seconds, np.reciprocal(sampling_frequency_hz) )
#print(time_vector)


vascular_latency_matrix[0,1] = vascular_latency_vector[1]
vascular_latency_matrix[1,2] = vascular_latency_vector[2]
vascular_latency_matrix[2,3] = vascular_latency_vector[3]
vascular_latency_matrix[3,4] = vascular_latency_vector[4]
vascular_latency_matrix[7,4] = vascular_latency_vector[4]
vascular_latency_matrix[0,5] = vascular_latency_vector[5]
vascular_latency_matrix[5,6] = vascular_latency_vector[6]
vascular_latency_matrix[6,7] = vascular_latency_vector[7]

vascular_coupling_matrix[0,1] = 0.74
vascular_coupling_matrix[1,2] = 0.9
vascular_coupling_matrix[2,3] = 0.7
vascular_coupling_matrix[3,4] = 0.83
vascular_coupling_matrix[7,4] = 0.94
vascular_coupling_matrix[0,5] = 0.6
vascular_coupling_matrix[5,6] = 0.8
vascular_coupling_matrix[6,7] = 0.52


# vascular_latency_vector is [739 495 391 178 581 263 516 503]