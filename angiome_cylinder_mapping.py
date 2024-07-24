# assign a separate binary mask to each segment
# conda activate vg

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:10:21 2019

@author: rdamseh
"""

# import skimage.io as skio
from matplotlib import pyplot as plt
import numpy as np
# import cv2
#import VascGraph as vg
#from tqdm import tqdm
import h5py
#import zarr
import multiprocessing
import seaborn as sns
import pandas as pd
import os


def save_dataset(save_file_name,group_name,dataset):
  root = zarr.open_group(save_file_name, mode='a')
  fill_me = root.require_group(group_name)
  root[group_name] = dataset


def h5_save_dataset(save_file_name,dataset_name,dataset):
    with h5py.File(save_file_name, 'a') as fout:
        fout.require_dataset(dtype=dataset.dtype,
                             compression="gzip",
                             chunks=True,
                             name=dataset_name,
                             shape=dataset.shape)
        fout[dataset_name][...] = dataset

def calc_vascular_angles(vascular_array):
    vascular_length = np.sqrt(np.sum(vascular_array**2, axis=0))
    vascular_length[vascular_length==0]=1.0 # avoid division by zero
    vascular_direction = np.abs(vascular_array.T)/vascular_length[:, None]
    return np.divide(180,np.pi) * np.arccos(vascular_direction)

def grid_plot(dataframe, file_name):
    t = sns.PairGrid(dataframe, height=3, diag_sharey=False)
    #t.map_upper(sns.scatterplot, hue = "mouse_name" , size = dataframe["radius"])
    t.map_upper(sns.scatterplot, size = dataframe["radius"], hue = dataframe["mouse_name"])
    t.map_lower(sns.kdeplot, shade=True)
    t.map_diag(sns.kdeplot, lw=2)
    t.savefig(file_name, dpi=100)
    print(file_name + ' ready (:')

def calc_proto_table(mouse_name, angiome_file_name, column_labels):
    angiome_handle = h5py.File(angiome_file_name, mode='r')
    print(angiome_handle.keys())
    vascular_radii = np.squeeze(angiome_handle['VesselRadii'])
    vascular_radii[vascular_radii == 0] = np.nan
    centerline_positions = angiome_handle['VesselCentroids']
    all_segment_edges = angiome_handle['all_segment_edges']

    num_vessel_segments = angiome_handle['num_vessel_segments']
    num_vessel_segments = np.uint64(num_vessel_segments[0,0])

    print('mouse name is: ' + mouse_name)
    

    #all_segment_edges = all_segment_edges.astype('int64')

    num_edges = all_segment_edges.shape[1]
    #num_edges = 100 # TEMPORARY

    first_interval_per_segment = np.uint64(np.squeeze(angiome_handle['index_first_interval_per_segment']))
    vessel_type_code = np.uint64(angiome_handle['vessel_type'])
    first_vertex_per_segment = np.uint64(angiome_handle['first_vertex_per_segment'])
    last_vertex_per_segment = np.uint64(angiome_handle['last_vertex_per_segment'])

    a = angiome_handle['vessel_type']
    print(a)
    print(f'shape of first_interval_per_segment is {first_interval_per_segment.shape}')
    print(vessel_type_code)
    print(f'shape of vessel_type_code is {vessel_type_code.shape}')
    
    '''
    print(f'first values of first_vertex_per_segment are {first_vertex_per_segment[:10]}')
    print(f'first values of last_vertex_per_segment are {last_vertex_per_segment[:10]}')
    print(f'first values of first_interval_per_segment are {first_interval_per_segment[:10]}')
    print(f'first values of all_segment_edges are {all_segment_edges[:10]}')
    print(f'first values of centerline_positions are {centerline_positions[:,:10]}')
    '''
    

    segment_indices = np.ones((num_edges,1), dtype='uint64')
    segment_type_code = np.ones((num_edges,1), dtype='uint64')

    #    (2, 132214)
    #(3, 684738)


    vascular_vector_array = np.zeros((3,num_edges))
    grand_vascular_vector_array = np.zeros((3,num_vessel_segments))
    depth_array = np.zeros((num_edges,1))
    grand_depth_array = np.zeros((num_vessel_segments,1))
    
    for edge_num in range(num_edges):
        #print(all_segment_edges[1,edge_num]-1)
        #print(str(centerline_positions[:,all_segment_edges[1,edge_num]-1] - centerline_positions[:,all_segment_edges[0,edge_num]-1]))
        vascular_vector_array[:,edge_num] = centerline_positions[:,all_segment_edges[1,edge_num]-1] - centerline_positions[:,all_segment_edges[0,edge_num]-1] # shifting from matlab indices to python indices
        depth_array[edge_num,0] = np.minimum(   centerline_positions[2,all_segment_edges[1,edge_num]-1] , centerline_positions[2,all_segment_edges[0,edge_num]-1] )
        #depth_array[edge_num,0] = np.minimum(   centerline_positions[1,all_segment_edges[1,edge_num]-1] , centerline_positions[1,all_segment_edges[0,edge_num]-1] )

        #rad_vec = [vascular_radii[all_segment_edges[1,edge_num]-1] ,  vascular_radii[all_segment_edges[0,edge_num]-1]]
        #edge_radii[edge_num] = np.nanmean(rad_vec)
    
    for segment_number, segment_index in enumerate(first_interval_per_segment[:-1]):
        next_segment_index = first_interval_per_segment[segment_number+1]
        #print(f'segment index and following segment index are {segment_index} and {next_segment_index}')
        segment_indices[segment_index:next_segment_index,0] *= segment_number
        segment_type_code[segment_index:next_segment_index,0] *= vessel_type_code[segment_number,0]
        #print(f'assigned segment indices are {segment_indices[segment_index:next_segment_index,0]}')
        #segment_indices[slice(segment_index,segment_index+1),0] *= segment_number
        try:
            grand_vascular_vector_array[:,segment_number] = centerline_positions[:,first_vertex_per_segment[segment_number,0]] - centerline_positions[:,last_vertex_per_segment[segment_number,0]] # shifting from matlab indices to python indices
        except:
            print(f'respective shapes are {first_vertex_per_segment.shape}, {last_vertex_per_segment.shape}, {centerline_positions.shape}')   
            print(f'segment number is {segment_number}')
            print(f'first_vertex_per_segment is {first_vertex_per_segment[segment_number,0]}')
            print(f'last_vertex_per_segment is {last_vertex_per_segment[segment_number,0]}')
            print(f'centerline positions are {centerline_positions[:,first_vertex_per_segment[segment_number,0]]}')
            print(f'centerline positions are {centerline_positions[:,last_vertex_per_segment[segment_number,0]]}')
            print(f'respective shapes are {first_vertex_per_segment.shape}, {last_vertex_per_segment.shape}, {centerline_positions.shape}')           
        grand_depth_array[segment_number,0] = np.minimum(   centerline_positions[2,first_vertex_per_segment[segment_number,0]] , centerline_positions[2,last_vertex_per_segment[segment_number,0]] )
        
    print('histogram of segment_type_code is:')
    print(np.histogram(segment_type_code))
    print('segment_indices looks like:')
    print(segment_indices)
    print('histogram of segment_indices looks like:')
    print(np.histogram(segment_indices))
    print(f'number of segment indices smaller than 2 is {np.sum(segment_indices<2)}')
    #vascular_vector_array = centerline_positions[:,all_segment_edges[1,:]-1] - centerline_positions[:,all_segment_edges[0,:]-1]
    print(all_segment_edges.dtype)
    '''
    for coor_num in range(3):
        vascular_vector_array[coor_num,:] = centerline_positions[coor_num,all_segment_edges[1,:].astype('uint64')-1] - centerline_positions[coor_num,all_segment_edges[0,:].astype('uint64')-1] # shifting from matlab indices to python indices
    '''
    rad_mat = np.stack( (vascular_radii[all_segment_edges[0,:].astype('uint64')-1], vascular_radii[all_segment_edges[1,:].astype('uint64')-1]), axis=1)
    grand_rad_mat = np.stack( (vascular_radii[first_vertex_per_segment], vascular_radii[last_vertex_per_segment]), axis=1)
    mean_rad = np.nanmean(rad_mat,axis=1)
    grand_mean_rad = np.nanmean(grand_rad_mat,axis=1)
    #edge_radii = (vascular_radii[all_segment_edges[1,:]-1] +  vascular_radii[all_segment_edges[0,:]-1]) / 2.0
    num_corrupt_radii = np.sum(np.isnan(rad_mat),axis=(0,1))
    print(f'the number of corrupt radii is {num_corrupt_radii}')

    



    expanded_radii = np.expand_dims(mean_rad,axis=1)

    vascular_angles = calc_vascular_angles(vascular_vector_array)
    grand_vascular_angles = calc_vascular_angles(grand_vascular_vector_array)
    
    #print(vascular_vector_array.shape)
    #print(vascular_vector_length.shape)
    #print(vascular_vector_direction.shape)
    #print(expanded_radii.shape)

    #proto_table =  np.concatenate((vascular_vector_direction, vascular_angles, expanded_radii, depth_array),axis=1)
    #column_labels = ['x_cosine', 'y_cosine', 'z_cosine','x_angle', 'y_angle', 'z_angle', 'radius', 'depth']

    proto_table =  np.concatenate((vascular_angles, expanded_radii, depth_array, segment_type_code),axis=1)
    grand_proto_table = np.concatenate((grand_vascular_angles, grand_mean_rad, grand_depth_array, vessel_type_code),axis=1)

    print('proto_table looks like:')
    print(proto_table)

    inner_column_labels = column_labels[:-2] # reserve the two right-most columns for mouse_name and segment_indices, or else an ugly bug will pop up ):

    df = pd.DataFrame(data=proto_table, columns=inner_column_labels)
    gdf = pd.DataFrame(data=grand_proto_table, columns=inner_column_labels)

    df['mouse_name'] = mouse_name
    df['segment_indices'] = segment_indices
    gdf['mouse_name'] = mouse_name
    gdf['segment_indices'] = np.arange(num_vessel_segments) # unlike df, gdf has exactly one entry per vessel segment
    return df, gdf



if __name__=='__main__':

    

    # binary image
    #segpath='D:\code\VascularGraph\synth1.mat'
    #segpath = 'binary_mask.mat'
    #seg=vg.GraphIO.ReadStackMat(segpath).GetOutput()

    # list of primes
    #list_of_primes_fname = 'D:/Documents/first_100000_primes.txt'
    #array_of_primes = np.ravel(np.loadtxt(list_of_primes_fname, dtype='uint64'))
    
    coarse_factor = str(5)
    fld = 'D:\Downloads'
    list_of_mouse_names = ['au', 'av', 'co', 'db']
    #list_of_mouse_names = ['au'] # for debug purposes

    
    # reserve the two right-most columns for mouse_name and segment_indices, or else an ugly bug will pop up ):
    outer_column_labels = ['x_angle', 'y_angle', 'z_angle', 'radius', 'depth', 'segment_type', 'mouse_name','segment_indices' ]
    df = pd.DataFrame(columns=outer_column_labels)
    gdf = pd.DataFrame(columns=outer_column_labels)
    #df['segment_type'] = df['segment_type'].astype('float')

    print(df)

    for mouse_num, mouse_name in enumerate(list_of_mouse_names):
        angiome_file_name = os.path.join(fld , 'segments_' + coarse_factor + '_' + mouse_name + '_to_pajek.mat')
        temp_df, temp_gdf = calc_proto_table(mouse_name, angiome_file_name, outer_column_labels)
        #temp_df['segment_indices'] += np.uint64(mouse_num*1000000)
        df = pd.concat([df,temp_df], ignore_index=True)
        gdf = pd.concat([gdf,temp_gdf], ignore_index=True)


    print(df)

    df_pa = df[(df.segment_type < 431) & (df.segment_type > 429)]
    gdf_pa = gdf[(gdf.segment_type < 431) & (gdf.segment_type > 429)]


    #df2 = df[['radius','z_angle','x_angle','mouse_name','segment_indices','z_cosine']]
    df2 = df[['depth','x_angle','y_angle','z_angle','radius','mouse_name','segment_indices']]

    gdf2 = gdf[['depth','x_angle','y_angle','z_angle','radius']]
    #min_df = df[['depth','y_angle']]

    
    #df['label'] = mouse_name
    print('still well after df')
    #
    #rdf = df.pivot_table(values = 'z_angle', columns=['radius'], aggfunc=(np.mean))


    sfn3 = angiome_file_name.replace('.mat' ,'_df.zip')
    sfn4 = angiome_file_name.replace('.mat' ,'_rdf.zip')
    sfn5 = angiome_file_name.replace('.mat' ,'_gdf.zip')



    #fig, ax = plt.subplots(nrows=1,ncols=1,figsize = [6, 6])

    sfn0 = angiome_file_name.replace('.mat' ,'_z_angle_vs_depth.png')
    sfn1 = angiome_file_name.replace('.mat' ,'_grand_z_angle_vs_depth.png')
    sfn02 = angiome_file_name.replace('.mat' ,'_z_angle_vs_depth_pa.png')  
    sfn12 = angiome_file_name.replace('.mat' ,'_grand_z_angle_vs_depth_pa.png')   

    print('df duplicates:')
    print(df[df.index.duplicated()])

    print('gdf duplicates:')
    print(gdf[gdf.index.duplicated()])

    grid_plot(df_pa, sfn02) 
    grid_plot(gdf_pa, sfn12)
    grid_plot(df, sfn0) 
    grid_plot(gdf, sfn1)

   
    df.to_pickle(sfn3)
    gdf.to_pickle(sfn5)   
    




    '''
    g0 = sns.PairGrid(min_df, height=5)

    g0.map_diag(sns.distplot)
    g0.map_lower(sns.regplot)
    g0.map_upper(sns.residplot)

    

    g0.savefig(sfn0, dpi=300)

    print('df regplot ready')
    '''



    



    


    '''
    hdf5_file_name = angiome_file_name.replace(".mat", ".hdf5")

    h5_save_dataset(hdf5_file_name, 'vascular_angles', vascular_angles)
    h5_save_dataset(hdf5_file_name, 'radii', mean_rad)
    h5_save_dataset(hdf5_file_name, 'grand_angles', grand_vascular_angles)
    '''    
    


    

    '''
    rdf = df2.pivot_table(values = ['x_angle','y_angle','z_angle'], index=['mouse_name','segment_indices'], columns=['depth','radius'], aggfunc=(np.mean))
    rdf.to_pickle(sfn4)
    sfn2 = angiome_file_name.replace('.mat' ,'_pivot_z_angle_scatterplot.png')
    grid_plot(rdf, sfn2)
    '''
    '''
    fig2, ax2 = plt.subplots(nrows=1,ncols=1,figsize = [6, 6])

    #g2 = sns.PairGrid(rdf, height=15)
    g2 = sns.PairGrid(rdf, height=5)

    g2.map_diag(sns.distplot)
    g2.map_lower(sns.regplot)
    g2.map_upper(sns.residplot)
    

    print('regplot ready')

    fld = 'D:/Downloads/'

    
    

    g2.savefig(sfn2, dpi=300)
    
    '''

    