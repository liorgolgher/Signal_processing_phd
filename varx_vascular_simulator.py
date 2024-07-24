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
import zarr
import numpy as np
from numcodecs import Blosc, Zstd 
from scipy import stats, signal
import matplotlib.pyplot as plt
import pickle
import os

rng = default_rng()

num_time_samples = 5000 # 3000#
num_vessels = 8 # start with a oversimplified network composed of 8 vessel segments and two bifurcations as depicted above. Blood flow in vessels 0,1,5 is measurable and is driven by blood flow in vessels 2,6 
sampling_frequency_hz = float(30) # 30 volumes per second smoothed to 10 samples per second
rbc_velocity = 4.5 # start with a constant velocity of red blood cells across the entire network. RBC transit time and inter-vascular latency are dictated by vascular distance d (t=d/v_rbc), with no dispersion.
heartbeat_frequency_hz = float(10)*0.01 # should be 10Hz ± 5Hz. Here changed to artificially lower values to satisfy sufficient sampling

num_training_samples = 2000
step_sample_time = 3000 # model a step function that starts at this sample

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




def save_dataset(save_file_name,group_name,dataset):
    root = zarr.open_group(save_file_name, mode='w')
    fill_me = root.require_group(group_name)
    root[group_name] = dataset

zarr.storage.default_compressor = Zstd(level=3)
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

vascular_lengths = np.abs(rng.normal(30, 10, size=(num_vessels,)))
#vascular_lengths = np.abs(np.random.normal(5, 1, size=(num_vessels,)))

print(f'vascular_lengths are {vascular_lengths}')

#vascular_latency_vector = np.rint((vascular_lengths*sampling_frequency_hz / rbc_velocity), dtype='int64')
vascular_latency_vector = np.rint(vascular_lengths * sampling_frequency_hz * np.reciprocal(rbc_velocity)).astype('int64')
print(f'vascular_latency_vector is {vascular_latency_vector}')

vascular_latency_matrix = np.ones((num_vessels,num_vessels), dtype='int64')
vascular_coupling_matrix = np.zeros((num_vessels,num_vessels))


### Build vascular relationship network
#       ___2___
#     1/       \ 3
#__0__/         \____4___
#     \         / 
#     5\___6___/ 7
#

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

vascular_coupling_matrix[0,1] = 0.4
vascular_coupling_matrix[1,2] = 0.9
vascular_coupling_matrix[2,3] = 0.7
vascular_coupling_matrix[3,4] = 0.3
vascular_coupling_matrix[7,4] = 0.4
vascular_coupling_matrix[0,5] = 0.6
vascular_coupling_matrix[5,6] = 0.8
vascular_coupling_matrix[6,7] = 0.2


sos = signal.butter(4,0.15, 'low', output='sos', fs=sampling_frequency_hz)

sos_broader = signal.butter(4,2, 'low', output='sos', fs=sampling_frequency_hz)


cbf = np.ones((num_vessels,time_vector.shape[0])) # each line is a vessel, each column is a step in time

raw_input_cbf = 1 + np.abs(rng.normal(50, 10, size=pre_filtered_time_vector.shape))

input_cbf = 3*signal.sosfilt(sos, raw_input_cbf) + signal.sosfilt(sos_broader, raw_input_cbf)

plt.plot(pre_filtered_time_vector,raw_input_cbf)
plt.plot(pre_filtered_time_vector,input_cbf)
plt.show()


#cbf[4,:] = np.sin(2*np.pi*heartbeat_frequency_hz*time_vector) + 2 
#cbf[4,:] += 0.4* (1 + np.random.randn(*cbf[4,:].shape))
cbf[4,:] = input_cbf[-len(time_vector):]
#cbf[4,step_sample_time:] += 0.2 * np.max(cbf[4,:])


cbf[3,:] = vascular_coupling_matrix[3,4] * np.roll(cbf[4,:], vascular_latency_matrix[3,4])
cbf[7,:] = vascular_coupling_matrix[7,4] * np.roll(cbf[4,:], vascular_latency_matrix[7,4])

cbf[2,:] = vascular_coupling_matrix[2,3] * np.roll(cbf[3,:], vascular_latency_matrix[2,3])
cbf[6,:] = vascular_coupling_matrix[6,7] * np.roll(cbf[7,:], vascular_latency_matrix[6,7])

#cbf[2,:]= np.random.randn(1,num_time_samples) * rng.random() + (10*rng.random()) 
#cbf[6,:]= np.random.randn(1,num_time_samples) * rng.random() + (10*rng.random())

cbf[1,:] = vascular_coupling_matrix[1,2] * np.roll(cbf[2,:], vascular_latency_matrix[1,2])
cbf[5,:] = vascular_coupling_matrix[5,6] * np.roll(cbf[6,:], vascular_latency_matrix[5,6])

c01 = vascular_coupling_matrix[0,1] * np.roll(cbf[1,:], vascular_latency_matrix[0,1])
c05 = vascular_coupling_matrix[0,5] * np.roll(cbf[5,:], vascular_latency_matrix[0,5])
cbf[0,:] = c01 + c05
plt.plot(time_vector, np.transpose(cbf))
plt.show()

filtered_cbf = signal.sosfilt(sos_broader, cbf, axis=1)
#filtered_cbf = cbf

#sos = signal.butter(4, 0.3, 'low', output='sos', fs=sampling_frequency_hz)


#fitted_vessel_indices = [0, 1, 5]
#fitted_vessel_indices = [1,2,3]
#fitted_vessel_indices = [0, 4]
fitted_vessel_indices = [0,1,2,3,4,5,6,7]


#filtered_cbf = cbf

minimal_lag = np.min(vascular_latency_vector[fitted_vessel_indices])
maximal_lag = np.max(vascular_latency_vector[fitted_vessel_indices])
print(f'minimal lag is {minimal_lag} samples')

start_from = minimal_lag + 100
end_at = num_training_samples # -100

#t_vec = np.linspace(0,num_time_samples/sampling_frequency_hz, num_time_samples)
cbf = cbf.T
filtered_cbf = filtered_cbf.T




used_cbf = filtered_cbf[start_from:end_at,fitted_vessel_indices]
short_time_vector = time_vector[start_from:end_at]
forecast_time_vector = time_vector[end_at] + time_vector

#plt.plot(t_vec[start_from:], cbf[start_from:,:])
plt.plot(short_time_vector,used_cbf)

plt.show()

print(f'The number of nan values in used_cbf is {np.sum(np.isnan(used_cbf[:]))}')

#data = list()
#data.append(cbf)

# print(data)

print(f'shape of used_cbf is {used_cbf.shape}')
print(used_cbf)

# fit model
model = VAR(used_cbf)
model_fit = model.fit(maxlags=int(maximal_lag*1.8))
# make prediction
print(model_fit.summary())
yhat = model_fit.forecast(model_fit.y, steps=num_time_samples)
#print(yhat)
plt.plot(time_vector[start_from:],filtered_cbf[start_from:,:],alpha=0.5)
bloody_end = -1000 # crop the last samples that look horrible
plt.plot(forecast_time_vector[:bloody_end],yhat[:bloody_end,:],alpha=0.5)
plt.show()

data_directory = 'D:' + os.sep + 'Downloads' 


pickle_fn = data_directory + os.sep + 'varx_model.p'
        
print('saving to ' + pickle_fn)

with open(pickle_fn, 'wb') as f:
    pickle.dump(model_fit, f, protocol=4)


print('all done')


