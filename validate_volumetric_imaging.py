# compare vascular/neuronal traces extracted from TAG-based imaging to planar imaging


import numpy as np 

import zarr
import os
import matplotlib.pyplot as plt
from skimage.transform import downscale_local_mean
import glob
import pandas as pd
import seaborn as sns
import multiprocessing
import statsmodels.formula.api as smf

#tz_downscale_factors = (15,31,1) # down-sample vessels over (time, z, number of vessels)
t_downscale_factors = (15,1) # down-sample vessels over (time, number of vessels) given an imaging rate of 30.04 vps


fld = os.path.join('D:' + os.sep, 'vvp') # CHANGE ME LATER
fn1 = fld + 'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_512l_1slow_3p5x_zoom_+100um_height_FOV1_036_vascular_traces_tz.zarr'
fn2 = fld + 'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_512l_1slow_3p5x_zoom_+200um_height_FOV1_TAG_OFF_SYNC_ON_+20um_deep_comparable_to_200um_108_vascular_traces_tz.zarr'


fwc1 = os.path.join(fld ,'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_512l_1slow_3p5x_zoom_+100um_height_FOV1_???_vascular_traces_tz.zarr')
fwc2 = os.path.join(fld , 'TLP1_Thy2_GCaMP7_FITC_mouse_930nm_512l_1slow_3p5x_zoom_+200um_height_FOV1_TAG_OFF_SYNC_ON_+20um_deep_comparable_to_200um_???_vascular_traces_tz.zarr')


def get_traces(input_list):
    num_data_files = len(input_list)
    '''
    
    print(input_list[0])
    print(input_list[0][0])
    print(input_list[0][1])
    print(input_list[0][2])
    print(input_list[0][3])   

    
    downscale_factors = input_list[0][2]
    
    print(downscale_factors)
    '''
    lines_per_file = int(input_list[0][-1])
    print(str(num_data_files) + ' files found')

    #fn_list = glob.glob(file_wc)
    #num_data_files = len(fn_list)
    #print(str(num_data_files) + ' files found')
    
    print("************* {}".format(type(input_list)))
    '''for out_index, inner_list in enumerate(input_list):
        print("1")
        print(type(inner_list))
        print("2")
        print(inner_list)
        print("3")'''

        
            

    #fn_list = [(idx, file_name, imaging_type) for inner_list in enumerate(input_list)]

    #z = zarr.open(fn_list[0],mode='r')
    #tmp = downscale_local_mean(np.array(z['vascular_dynamics_1500']), downscale_factors)
    # order of dimensions is (time, number of vessels)
    #num_time_samples, num_vessels = tmp.shape
    # Datatable is a long-form array - soon to be DF - which contains
    # the measurement value per file per vessel per timepoint. It also
    # has a label column, marking TAG on\off status. The last column
    # is the vascular volume
    # data columns are dataset_number, imaging_type, vessel_number, timestep_number, vessel_volume
    datatable = np.zeros((num_data_files * lines_per_file, 5))
    #all_traces = np.zeros((num_time_samples,num_vessels,num_data_files))
    #samples_per_file = num_time_samples * num_vessels
    #print(fn_list[0])

    '''

    tmp_ts = np.mgrid[0:100,0:201]
    tmp_vn = np.mgrid[0:201,0:100]
    timestep_indices = tmp_ts[0,:,:]
    timestep_indices = timestep_indices.T.ravel()
    vessel_numbers = tmp_vn[0,:,:]
    vessel_numbers = vessel_numbers.ravel()
    '''

    for file_num, (file_name, imaging_type, downscale_factors, vessel_numbers, timestep_indices, lines_per_file) in enumerate(input_list):
        #vessel_number, timestep_number    
        print("{}".format(file_name))
        z = zarr.open(file_name, mode='r')
        downscaled = downscale_local_mean(np.array(z['vascular_dynamics_1500']), downscale_factors)
        #       
        #downscaled = np.random.rand(100,201) # DUMMY DATA

        downscaled = downscaled.T.ravel()
        print(downscaled.shape)
        type(downscaled)
        datatable[file_num * lines_per_file:(file_num + 1) * lines_per_file, 0] = file_num
        datatable[file_num * lines_per_file:(file_num + 1) * lines_per_file, 1] = imaging_type
        datatable[file_num * lines_per_file:(file_num + 1) * lines_per_file, 2] = vessel_numbers
        datatable[file_num * lines_per_file:(file_num + 1) * lines_per_file, 3] = timestep_indices
        
        datatable[file_num * lines_per_file:(file_num + 1) * lines_per_file, 4] = downscaled # actual vessel volume per vessel segment per time point
    return datatable
'''
    for fnum, fname, imaging_type in fn_list:
    #for fnum, fname in enumerate(fn_list):
        #z = zarr.open(fname,mode='r')
        #downscaled = downscale_local_mean(np.array(z['vascular_dynamics_1500']), downscale_factors)
        downscaled = np.rand.random((100,201))
        downscaled = downscaled.T.ravel()
        datatable[fnum * lines_per_file:(fnum + 1) * lines_per_file, 5] = downscaled


        #all_traces[:,:,fnum] = downscaled
'''




'''
print(z.keys())

tz = np.array(z['vascular_dynamics_tz_1500'])
dtz = downscale_local_mean(tz, downscale_factors)
print(dtz.shape)

vessel_num = 1
plt.plot(dtz[:,:,vessel_num])
plt.show()

tag_on_traces   =    get_traces(fwc1, 'volumetric', t_downscale_factors)
tag_off_traces  =    get_traces(fwc2, 'planar', t_downscale_factors)

def grand_average_traces(traces):
    gav = np.mean(traces,axis=(0,2))
    std = np.std(traces,axis=(0,2), ddof=1)
    return gav, std

def average_traces(traces):
    avg = np.mean(traces,axis=0)
    print(avg.shape)
    std_within = np.std(avg, axis=1, ddof=1)
    print(std_within.shape)
    return avg, std_within

avg_on, stdw_on     = average_traces(tag_on_traces)
avg_off, stdw_off   = average_traces(tag_off_traces)

print(stdw_on)
print(stdw_off)

gav_on, gst_on      = grand_average_traces(tag_on_traces)
gav_off, gst_off    = grand_average_traces(tag_off_traces)

fig1, ax1 = plt.subplots()

for filenum in range(avg_off.shape[1]):
    ax1.scatter(np.arange(0,201), avg_off[:,filenum], color='blue', alpha=0.15)

for filenum in range(avg_on.shape[1]):
    ax1.scatter(np.arange(0,201), avg_on[:,filenum], color='red', alpha=0.15)

ax1.set_xlabel('Vessel segment #')

ax1.title.set_text('TAG ON (red) vs. TAG OFF (blue) volume for 201 vessel segments')


#plt.scatter(np.arange(0,201), 100*gav_on/gav_off)

fig2, (ax2,ax3) = plt.subplots(nrows=1,ncols=2)

ax2.errorbar(gav_off, gav_on, xerr=gst_off, yerr=gst_on, fmt='.')

ax2.title.set_text('TAG ON vs. TAG OFF volume for 201 vessel segments')
ax2.set_xlabel('TAG OFF volume')
ax2.set_ylabel('TAG ON volume')

ax3.violinplot(100*gav_on/gav_off, showmedians=True,points=200)

ax3.title.set_text('TAG ON / TAG OFF ratios for 201 vessel segments [%]')
'''


num_frames = int(900)
num_vessels = int(72)
ds_num_time_samples = int( num_frames / t_downscale_factors[0])
lpf = ds_num_time_samples * num_vessels # number of lines per file is 1500 time samples * 201 vessel segments

tmp_ts = np.mgrid[0:ds_num_time_samples,0:num_vessels]
tmp_vn = np.mgrid[0:num_vessels,0:ds_num_time_samples]
timestep_indices = tmp_ts[0,:,:]
timestep_indices = timestep_indices.T.ravel()
vessel_numbers = tmp_vn[0,:,:]
vessel_numbers = vessel_numbers.ravel()

finput0 = [(fname, 0, t_downscale_factors, vessel_numbers, timestep_indices, lpf) for fname in glob.glob(fwc2)] #planar
finput1 = [(fname, 1, t_downscale_factors, vessel_numbers, timestep_indices, lpf) for fname in glob.glob(fwc1)] #volumetric

all_inputs = finput0 + finput1

all_traces = get_traces(all_inputs)

print(all_traces.shape)
print(all_traces[4000:4003,:])

column_labels = ['dataset', 'imaging type', 'vessel', 'timestep', 'vascular volume']

df = pd.DataFrame(data=all_traces, columns=column_labels)
df["dataset"] = df['dataset'].astype('int')
df["vessel"] = df['vessel'].astype('int')
df["timestep"] = df['timestep'].astype('int')
df["imaging type"] = df['imaging type'].astype('int')


#rdf = df.pivot_table(values = 'vascular volume', index=['vessel'], columns=['imaging type'], aggfunc=(np.mean, np.std))
rdf = df.pivot_table(values = 'vascular volume', index=['vessel'], columns=['imaging type'], aggfunc=(np.mean))
rdf.rename(columns={0:'Planar',
                          1:'Volumetric'}, 
                 inplace=True)

rdf = rdf.rename_axis('arbitrary_vessel_index').sort_values(by = ['Planar'])
rdf['ordinal_vessel_index'] = np.arange(num_vessels)


df = df.merge(rdf, left_on='vessel', right_on='arbitrary_vessel_index')


print(df)

reshaped_df = df.pivot_table(values = 'vascular volume', index=['ordinal_vessel_index'], columns=['imaging type','dataset'], aggfunc=(np.mean, np.std)) # calculate the standard deviation separately for each dataset, then find the average std across datasets


cv0 = 100 * reshaped_df['std'][0].mean(axis=1) / reshaped_df['mean'][0].mean(axis=1)
cv1 = 100 * reshaped_df['std'][1].mean(axis=1) / reshaped_df['mean'][1].mean(axis=1)

cv_dict = {'Planar': cv0, 
        'Volumetric': cv1}

cv_df = pd.DataFrame(cv_dict)

# print(cv_df)

fig, ax = plt.subplots(nrows=1,ncols=1,figsize = [64, 4.8])
ax = sns.violinplot(x="ordinal_vessel_index", y="vascular volume", hue="imaging type",
                    data=df, palette="muted", split=True, scale="count", inner="quartile")



ax.title.set_text('Planar (blue) vs. Volumetric (red) of vessel volume for ' + str(int(num_vessels)) + ' vessel segments.\n Timesteps of ' + str(t_downscale_factors[0]/30) + ' seconds')

sfn = os.path.join(fld ,'violin_plot.png')
fig.savefig(sfn, dpi=500)






fig2, ax2 = plt.subplots(nrows=1,ncols=1,figsize = [12, 12])

#ax2 = sns.scatterplot(x=0, y=1, data=reshaped_df['vascular volume'], alpha=0.5, s=10)
#sfn2 = os.path.join(fld ,'median_scatterplot.png')
#ax2.title.set_text('Median Planar vs. Volumetric estimate of vessel volume for ' + str(int(num_vessels)) + ' vessel segments.\n Timesteps of ' + str(t_downscale_factors[0]/30) + ' seconds')


#ax2 = sns.regplot(x='Planar', y='Volumetric', data=cv_df)

g2 = sns.PairGrid(cv_df, height=15)

g2.map_diag(sns.distplot)
g2.map_lower(sns.regplot)
g2.map_upper(sns.residplot)

sfn2 = os.path.join(fld ,'CV_scatterplot.png')
#ax2.title.set_text('Coefficient of variation for planar and volumetric estimates of vessel volume for ' + str(int(num_vessels)) + ' vessel segments.\n Timesteps of ' + str(t_downscale_factors[0]/30) + ' seconds')
#ax2.set_xlabel('Planar coefficient of variation [%]')
#ax2.set_ylabel('Volumetric coefficient of variation [%]')

g2.savefig(sfn2, dpi=500)

regression_thingy = smf.ols(formula="Volumetric ~ Planar", data=cv_df).fit()
print(regression_thingy.summary())
print('Parameters: ', regression_thingy.params)
print('R2: ', regression_thingy.rsquared)



fig5, ax5 = plt.subplots(nrows=1,ncols=1,figsize = [12, 12])

print('rdf')
print(rdf)
#ax5 = sns.lmplot(x='Planar', y='Volumetric', data=rdf, alpha=0.4, size=5)
#ax5 = sns.jointplot(x='Planar', y='Volumetric', data=rdf, kind="kde")

g = sns.PairGrid(rdf.drop(columns=['ordinal_vessel_index']), height=15)

g.map_diag(sns.distplot)
g.map_lower(sns.regplot)
g.map_upper(sns.residplot)

sfn5 = os.path.join(fld ,'mean_scatterplot.png')
#ax5.title.set_text('Mean Planar vs. Volumetric estimate of vessel volume for ' + str(int(num_vessels)) + ' vessel segments.\n Timesteps of ' + str(t_downscale_factors[0]/30) + ' seconds')
#fig5.savefig(sfn5, dpi=500)
#g.title.set_text('Mean Planar vs. Volumetric estimate of vessel volume for ' + str(int(num_vessels)) + ' vessel segments.\n Timesteps of ' + str(t_downscale_factors[0]/30) + ' seconds')
#title_list = ["apple", "banana", "cherry", "distribution"]
#g.set_titles(title_list)
g.savefig(sfn5, dpi=500)

#rdf.columns = rdf.columns.astype(str)

mean_regression_thingy = smf.ols(formula="Volumetric ~ Planar", data=rdf).fit()
print(mean_regression_thingy.summary())
print('Parameters: ', mean_regression_thingy.params)
print('R2: ', mean_regression_thingy.rsquared)