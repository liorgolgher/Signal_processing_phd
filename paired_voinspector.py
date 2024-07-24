# use pandas environment

from IPython.display import Image
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
import zarr
from numcodecs import Blosc, Zstd
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename
import os
import glob

# initializations:
zarr.storage.default_compressor = Zstd(level=3)
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

fld_name = '/data/Lior/lst/2020/2020_01_13/ttz/'

num_inspected_list_files = 1

# functions:

#data = np.random.randn(100, 3)
#df = pd.DataFrame(data, columns=['cell 1', 'cell 2', 'cell 3'])

file_wild_card = fld_name + 'T*_dff.zarr'
fld = '/data/Lior/lst/2020/2020_01_13/ttz/'

def dff_range_extractor(zarr_file_wild_card = None, home_folder='/data/Lior/lst/2020/2020_01_13/ttz/', askuser = 'Choose zarr file with dff data'):
    if zarr_file_wild_card is None:
        zarr_file_wild_card = askdirectory(initialdir=home_folder, title=askuser) # show an "Open" dialog box and return the path to the selected file
    print(zarr_file_wild_card)
    fn_list = glob.glob(zarr_file_wild_card)
    num_data_files = len(fn_list)

    print(str(num_data_files) + ' files found')

    z = zarr.open(fn_list[0],mode='r')

    num_cells = z['neuronal_dff'].shape[0]

    dff_range = np.zeros((num_data_files,num_cells))
    spike_rate = np.zeros((num_data_files,num_cells))
    imaging_rate = np.zeros((num_data_files))
    zarr_file_number = np.zeros((num_data_files))
    control_dataset_flag = np.zeros((num_data_files))

    for fnum, fname in enumerate(fn_list):
        print(fname)
        z = zarr.open(fname,mode='r')
        dff_array = np.array(z['neuronal_dff'])
        # 95th-5th percentile range is used as a crude measure of the dF/F SBR
        dff_range[fnum, :] = 100 * (np.percentile(dff_array, 95, axis=1) - np.percentile(dff_array, 5, axis=1))
        spike_rate[fnum, :] = np.array(z['spike_rate'])
        imaging_rate[fnum] = np.array(z['imaging_rate'])
        zarr_file_number[fnum] = int(fname[-26:-23])
        if fname[0] == 'C': # if dff dataset drawn from control VOIs with no neurons
            control_dataset_flag[fnum] = 1
    return dff_range, spike_rate, imaging_rate, zarr_file_number, control_dataset_flag

# dataframe construction:

dff_ranges, spike_rates, imaging_rates, zarr_file_numbers, control_dataset_flags = dff_range_extractor(zarr_file_wild_card = file_wild_card, home_folder=fld)

long_zarr_file_numbers = 1000*control_dataset_flags + zarr_file_numbers
file_labels = ['f#' + str(x) for x in long_zarr_file_numbers]
file_labels = ['0','21','67C','71C','201','211','466V']
cell_labels = ['cell #' + str(x) for x in range(dff_ranges.shape[1])]

print(dff_ranges.shape)

df = pd.DataFrame(data=dff_ranges.T,  index=cell_labels, columns=file_labels)
dfs = pd.DataFrame(data=spike_rates.T,  index=cell_labels, columns=file_labels)
# to extract cell 40 at the 1000th timepoint type df.iloc[40].iloc[1000]
#mdf = pd.melt(df)

print(df.shape)
#print(mdf.shape)



#mdf = pd.melt(other_df)
#df = pd.MultiIndex.from_arrays(fast_dff_dataset, names=range(fast_dff_dataset.shape[0]))

#sns.swarmplot(x="variable", y="value", hue="cell_labels", palette=["r", "c", "y"], data=mdf)
#sns.swarmplot(x="variable", y="value", palette=["r", "c", "y"], data=mdf)


# visualization:

#df['time'] = np.arange(100)

#df.iloc[:, :3].plot()

#df = df.drop('time', axis=1)

#df.melt()

#sns.barplot(data=mdf, x='variable', y='value')
#


#sns.set(style="ticks")
sns.set(style="ticks", color_codes=True)
#sns.pairplot(df.T, hue="file_labels")
#sns.pairplot(df.T)
#sns.pairplot(mdf)

#sns_plot = sns.pairplot(df, height=2.0)
sns_plot = sns.pairplot(dfs, height=2.0)
sns_plot.savefig(fld + 'pairplot_spike_rates.png', dpi=500)

plt.clf() # Clean parirplot figure from sns
#Image(filename='pairplot.png') # Show pairplot as image
#plt.show()
# see https://seaborn.pydata.org/examples/scatterplot_matrix.html

def dff_extractor(zarr_file_name = None, home_folder='/data/Lior/lst/2020/2020_01_13/ttz/', askuser = 'Choose zarr file with dff data'):
    if zarr_file_name is None:
        zarr_file_name = askdirectory(initialdir=home_folder, title=askuser) # show an "Open" dialog box and return the path to the selected file
    print(zarr_file_name)
    zarr_file_number = zarr_file_name[-26:-23]

    dff_dataset = np.array(zarr.open(zarr_file_name + '/neuronal_dff'))
    time_vector = np.array(zarr.open(zarr_file_name + '/time_vector'))
    imaging_rate = np.array(zarr.open(zarr_file_name + '/imaging_rate'))
    return dff_dataset, time_vector, imaging_rate, zarr_file_number
