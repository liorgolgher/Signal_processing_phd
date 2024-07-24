import zarr
import os
import matplotlib.pyplot as plt
import numpy as np

data_folder_name = 'D:\Downloads'

for filenum in range(2):
    fn = data_folder_name + os.sep + '2v' + str(filenum+1) + 'n.tif'


    load_fn =  fn.replace(".tif", "_milked.zarr")
    hdf5_file_name = load_fn.replace("zarr", "hdf5")

    z = zarr.open(load_fn,'r')

    print(z.keys())

    vascular_angle = np.array(z['vascular_angle'])
    print(f'vascular angle is {vascular_angle}')

    time_varying_vascular_diameter = np.array(z['time_varying_vascular_diameter'])
    branching_vessel_photon_count = np.array(z['branching_vessel_photon_count'])

    plt.plot(time_varying_vascular_diameter)
    plt.plot(np.divide(branching_vessel_photon_count,np.max(branching_vessel_photon_count))*(2+np.max(time_varying_vascular_diameter)))
    plt.show()

