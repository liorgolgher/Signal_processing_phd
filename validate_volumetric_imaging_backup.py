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


num_frames = int(1500)
num_vessels = int(201)
ds_num_time_samples = int( 1500 / t_downscale_factors[0])
lpf = ds_num_time_samples * num_vessels # number of lines per file is 1500 time samples * 201 vessel segments

tmp_ts = np.mgrid[0:ds_num_time_samples,0:num_vessels]
tmp_vn = np.mgrid[0:num_vessels,0:ds_num_time_samples]
timestep_indices = tmp_ts[0,:,:]
timestep_indices = timestep_indices.T.ravel()
vessel_numbers = tmp_vn[0,:,:]
vessel_numbers = vessel_numbers.ravel()

finput0 = [(fname, 0, t_downscale_factors, vessel_numbers, timestep_indices, lpf) for fname in glob.glob(fwc2)] #planar
finput1 = [(fname, 1, t_downscale_factors, vessel_numbers, timestep_indices, lpf) for fname in glob.glob(fwc1)] #volumetric
#print(finput1)
all_inputs = finput0 + finput1
#print(all_inputs)

#with multiprocessing.Pool() as mp:
#    mp.starmap(get_traces, all_inputs)

all_traces = get_traces(all_inputs)

print(all_traces.shape)
print(all_traces[40000:40003,:])

column_labels = ['dataset', 'imaging type', 'vessel', 'timestep', 'vascular volume']

#cell_labels = ['cell #' + str(x) for x in range(dff_ranges.shape[1])]

#print(dff_ranges.shape)

df = pd.DataFrame(data=all_traces, columns=column_labels)
df["dataset"] = df['dataset'].astype('int')
df["vessel"] = df['vessel'].astype('int')
df["timestep"] = df['timestep'].astype('int')
df["imaging type"] = df['imaging type'].astype('int')


def literal_imaging_type(TAG_status):
    if TAG_status:
        return 'volumetric'
    else:
        return 'planar'

'''
df0 = df["imaging type"=0]
df1 = df["imaging type"=1]

merged_df = df0
merged_df.merge(df1, on='imaging type' suffixes=('_planar', '_volumetric'))

print(merged_df)
'''

#df["imaging modality"] = literal_imaging_type(df['imaging type'])

#grouped = df.groupby('imaging type')

#grouped = df.groupby('vessel', literal_imaging_type('imaging type'))
#grouped = df.groupby(['vessel', 'imaging type'], axis='columns')


###


""" 
grouped = df.groupby(by=['imaging type','vessel'], as_index=False)
#grouped = df.groupby(by=['imaging type'], as_index=False)
#two_measures_per_obs = pd.concat([group for (name, group) in grouped if name in ['imaging type', 'vessel']])
two_measures_per_obs = grouped.get_group((0,0))
print('meme')
print(grouped.get_group((1,0)).loc[:, "vascular volume"])
two_measures_per_obs['volumetric'] = grouped.get_group((1,0)).loc[:, "vascular volume"]
for vn in range(1,num_vessels):
    two_measures_per_obs.append(grouped.get_group((0,vn)))
    two_measures_per_obs['volumetric'].append(grouped.get_group((1,vn)).loc[:, "vascular volume"])
#two_measures_per_obs = grouped.get_group((0,20))
two_measures_per_obs.rename({'vascular volume': 'planar'}, inplace=True, axis=1)
print('planar')
print(two_measures_per_obs)
'''
two_measures_per_obs['volumetric'] = grouped.get_group((1,0)).loc[:,"vascular volume"]
for vn in range(1,num_vessels):
    #print(str(vn))
    two_measures_per_obs['volumetric'].append(grouped.get_group((1,vn)).loc[:, "vascular volume"])
'''

#two_measures_per_obs['volumetric'] = grouped.get_group((1,0)).loc[:, "vascular volume"])
'''
for vn in range(5):
#    print(grouped.get_group((1,vn)).loc[:, "vascular volume"])
    two_measures_per_obs['volumetric'].append(grouped.get_group((1,vn)).loc[:, "vascular volume"])
'''

#for vn in range(1,num_vessels):
#    two_measures_per_obs['volumetric'].append(grouped.get_group((1,vn)).loc[:, "vascular volume"])

print('merged')
print(two_measures_per_obs)
'''    
two_measures_per_obs['volumetric'] = grouped.get_group((1,))["vascular volume"]
two_measures_per_obs['volumetric'] = grouped.get_group((1,)).loc[:, "vascular volume"]
two_measures_per_obs = two_measures_per_obs.rename({'vascular volume': 'planar'}, axis=1)
'''

fig2, ax2 = plt.subplots(nrows=1,ncols=1,figsize = [12, 12])

fig2 = sns.lmplot(x='planar', y='volumetric', data=two_measures_per_obs, scatter_kws={'alpha': 0.4, 's': 8})
ax2.set_aspect('equal')
#ax2 = sns.scatterplot(data=median_volume)

sfn2 = os.path.join(fld ,'lmplot_modality_vessel.png')
fig2.savefig(sfn2, dpi=500)
 """

#grouped = df.groupby(by=['imaging type'], as_index=False)
grouped = df.groupby(by=['imaging type','vessel'], as_index=False)
#print(grouped.groups)
pmpo = grouped.get_group((0,0))
vmpo = grouped.get_group((1,0))
for vn in range(1,num_vessels):
    pmpo.append(grouped.get_group((0,vn)))
    vmpo.append(grouped.get_group((1,vn)))

pmpo.rename({'vascular volume': 'planar'}, inplace=True, axis=1)
#pmpo.drop(columns=['dataset','imaging type'], inplace=True)

vmpo.rename({'vascular volume': 'volumetric'}, inplace=True, axis=1)
#vmpo.drop(columns=['dataset','imaging type'], inplace=True)

#print(vmpo)



mdf3 = pd.merge(pmpo, vmpo)

#mdf2 = mdf.copy()
#mdf2.drop(columns=['dataset','imaging type'], inplace=True)

print('merged_inner')
print(mdf3)

fig4, ax4 = plt.subplots(nrows=1,ncols=1,figsize = [12, 12])

fig4 = sns.lmplot(x='planar', y='volumetric', data=mdf3, scatter_kws={'alpha': 0.4, 's': 8})
ax4.set_aspect('equal')
sfn4 = os.path.join(fld ,'lmplot_by_vessel_and_type.png')
fig4.savefig(sfn4, dpi=500)


###


#print(grouped.groups[:,0])

#print(grouped.groups)

#print(grouped.indices)

#print(grouped["vascular volume"].indices)
#print(df.describe())


#grouped = df.groupby(['vessel', 'imaging type'], axis='rows')

reshaped_df = df.pivot_table(index='vessel', columns='imaging type', aggfunc=np.mean)
#var_df = df.pivot_table(index='vessel', columns='imaging type', aggfunc=np.var)
rdf = df.pivot_table(index='vessel', columns='imaging type')
#print(grouped.describe())




#median_volume = grouped.median()

#print(reshaped_df)
#print(rdf)
#print(grouped.describe())
#print(median_volume.describe())
#print(median_volume)
#print(median_volume.shape)
#print(median_volume.keys())



#df["imaging_modality"] = 'planar'
#cond = [df["imaging_type"] == 1]
#df["imaging_modality"][cond] = 'volumetric'

'''
vmin = 0
vmax = 25

cond = df['vessel']
df2 = df[df['vessel'] > vmin & df['vessel'] < vmax]
df2 = df[df['vessel'] > vmin & df['vessel'] < vmax]
'''


fig, ax = plt.subplots(nrows=1,ncols=1,figsize = [64, 4.8])
ax = sns.violinplot(x="vessel", y="vascular volume", hue="imaging type",
                    data=df, palette="muted", split=True, scale="count", inner="quartile")



#sns_plot = sns.pairplot(df, height=2.0)

#fig = ax.get_figure()



ax.title.set_text('Planar (blue) vs. Volumetric (red) of vessel volume for ' + str(int(num_vessels)) + ' vessel segments.\n Timesteps of ' + str(t_downscale_factors[0]/30) + ' seconds')

sfn = os.path.join(fld ,'violin_plot.png')
fig.savefig(sfn, dpi=500)



#fig2 = sns.lmplot(x=0, y=1, data=reshaped_df['vascular volume'], scatter_kws={'alpha': 0.4, 'size': 10})

#fig2 = sns.lmplot(x=grouped.groups[:,0], y=grouped.groups[:,1], data=grouped, scatter_kws={'alpha': 0.4, 'size': 10})
#fig2 = sns.lmplot(x=grouped.groups[0], y=grouped.groups[1], data=grouped, scatter_kws={'alpha': 0.4, 'size': 10})


#ax2.title.set_text('Volumetric vs. planar vascular cross-section for ' + str(int(num_vessels)) + ' vessel segments.\n Timesteps of ' + str(t_downscale_factors[0]/30) + ' seconds')
#fig2.axes.set_title("THANKS HAGAI!",fontsize=100)
#ax2.set_xlabel('Planar cross-section')
#ax2.set_ylabel('Volumetric cross-section')




fig2, ax2 = plt.subplots(nrows=1,ncols=1,figsize = [12, 12])

ax2 = sns.scatterplot(x=0, y=1, data=reshaped_df['vascular volume'], alpha=0.5, s=10)


ax2.set_aspect('equal')
#ax2 = sns.scatterplot(data=median_volume)

sfn2 = os.path.join(fld ,'median_scatterplot.png')
ax2.title.set_text('Median Planar vs. Volumetric estimate of vessel volume for ' + str(int(num_vessels)) + ' vessel segments.\n Timesteps of ' + str(t_downscale_factors[0]/30) + ' seconds')
fig2.savefig(sfn2, dpi=500)


'''
ax3 = sns.scatterplot(x=0, y=1, data=var_df['vascular volume'], alpha=0.5, s=10)
ax3.title.set_text('Variance in volumetric vs. planar vascular cross-section for ' + str(int(num_vessels)) + ' vessel segments.\n Timesteps of ' + str(t_downscale_factors[0]/30) + ' seconds')
ax3.set_xlabel('Variance of planar cross-section')
ax3.set_ylabel('Variance of volumetric cross-section')
'''




#plt.show()


'''
all_traces = get_traces(all_inputs)



plt.show()
'''



# PASTED FROM HERE:


grouped = df.groupby(by=['imaging type'], as_index=False)
two_measures_per_obs = grouped.get_group(0)
two_measures_per_obs.rename({'vascular volume': 'planar'}, inplace=True, axis=1)
two_measures_per_obs.drop(columns=['dataset','imaging type'], inplace=True)

vmpo = grouped.get_group(1)
vmpo.rename({'vascular volume': 'volumetric'}, inplace=True, axis=1)
vmpo.drop(columns=['dataset','imaging type'], inplace=True)


#two_measures_per_obs['volumetric'] = grouped.get_group(1).loc[:, "vascular volume"]
print(vmpo)

mdf = pd.merge(two_measures_per_obs, vmpo, how='outer')
mdf2 = mdf.copy()
#mdf2.drop(columns=['dataset','imaging type'], inplace=True)

print('merged_outer')
print(mdf2)

fig3, ax3 = plt.subplots(nrows=1,ncols=1,figsize = [12, 12])

fig3 = sns.lmplot(x='planar', y='volumetric', data=mdf2, scatter_kws={'alpha': 0.4, 's': 8})
ax3.set_aspect('equal')
sfn3 = os.path.join(fld ,'lmplot_by_type.png')
fig3.savefig(sfn3, dpi=500)
