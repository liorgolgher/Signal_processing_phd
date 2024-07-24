#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:10:21 2019

@author: rdamseh
"""

import skimage.io as skio
from matplotlib import pyplot as plt
import numpy as np
import cv2
import VascGraph as vg
from tqdm import tqdm
import os
  

class CreateCylinderMappings:
    
    '''
    This class create 3D maps based on oriented cylinders built at each graph edge 
    '''
        
    
    def __init__(self, g, im_size=None, to_return=['binary', 
                                                     'velocity', 
                                                     'so2', 
                                                     'hct', 
                                                     'gradient', 
                                                     'propagation',
                                                     'segments']):
                        
        self.g=g
        self.im_size=im_size
        self.GetImSize()
        
        # set the needed outputs
        self.tags={'binary':1,
                    'velocity':0, 
                    'so2':0, 
                    'hct':0, 
                    'gradient':0, 
                    'propagation':0,
                    'segments':0}
        
        for i in to_return:
            self.tags[i]=1
        
    def GetImSize(self):
        
        # shift graph geometry to start from zero coordinates
        if self.im_size is None:
            pos=np.array(self.g.GetNodesPos())
            pos=pos-np.min(pos, axis=0)[None, :]
            self.g.SetNodesPos(pos)
        
        # fixing radii below 2.0
        min_rad=2.0
        rad=np.array(self.g.GetRadii())
        ## MODIFIED BY LG 2020_06_14
        rad =  rad[np.isfinite(rad)]
        ## END OF MODIFICATION BY LG 2020_06_14
        
        rad[rad<min_rad]=min_rad
        self.g.SetRadii(rad)
        
        # get image size to be constructed
        if self.im_size is None:
            pos=np.array(self.g.GetNodesPos())
            real_s = np.max(pos, axis=0) # real image size
            new_s=real_s   
        else:
            new_s=real_s=self.im_size
            
        # shifting graph nodes to start from maxr
        pos=np.array(self.g.GetNodesPos())   
        ## MODIFIED BY LG 2020_06_14
        #maxr=np.max(rad)
        maxr=np.nanmax(rad)

        print('Max radius: '+str(maxr))
        print(maxr.dtype)
        ## END OF MODIFICATION BY LG 2020_06_14
        pos=pos+maxr
        self.g.SetNodesPos(pos) 

        # reconstructed image size after padding from two sides
        new_s=tuple((np.ceil(new_s+(2*maxr))).astype(int)) 
                    
        self.real_s = real_s
        self.new_s = new_s
        self.niter = self.g.number_of_edges()
        self.ends=np.ceil([[maxr, -maxr], 
                            [maxr, -maxr], 
                            [maxr, -maxr]]).astype(int)            
                
            
        print('Image size: '+str(self.new_s)) 
        
    def cylinder(self, direction, radius, length):
        
        '''
        Create a image cylinder  
        '''
        
        r=length+2*radius
        r=int(r)
        
        #print('r value', r)
        xrange, yrange, zrange = np.meshgrid(np.arange(-r, r+1),
                                             np.arange(-r, r+1),
                                             np.arange(-r, r+1), indexing='ij')
        size=np.shape(xrange)
        
        direction=direction.astype(float)
        va=np.sqrt((direction**2).sum())
        vnorm=direction/va
        
        p=np.array([xrange.ravel(), yrange.ravel(), zrange.ravel()]).T
        p=p.astype(float)
        amp=np.sqrt(np.sum(p**2, axis=1))
        amp[amp<1]=1
        
        cos=np.abs(np.sum(p*vnorm, axis=1)/amp)
        cos[cos>1]=1
        sin=np.sqrt(1-cos**2)
    
        shape0=(amp*sin)<radius # radius constrain
        shape1=(amp*cos<length) # length constrain
        
        a1=amp*cos-length
        a2=amp*sin
        shape2=(((a1**2+a2**2)**0.5)<(radius)) # rounded end constrain
        
        shape=shape0*(shape2+shape1)
        
        shape=np.reshape(shape, xrange.shape)
        c0 = np.where(shape)
        
        dot=np.sum(p*vnorm, axis=1)
        dot=((dot-dot.min())/(dot.max()-dot.min()))
        shape=shape*dot.reshape(shape.shape)
        
        return c0, size   


    def get_cylinder_infos(self, g, radius_scaling=None):
        
        info=dict()
        
        if self.tags['binary']:

            e=g.GetEdges()
            pos1=np.array([g.node[i[0]]['pos'] for i in e])
            pos2=np.array([g.node[i[1]]['pos'] for i in e])
            
            radius1=np.array([g.node[i[0]]['r'] for i in e])
            radius2=np.array([g.node[i[1]]['r'] for i in e]) 
            radius=(radius1+radius2)/2.0# radius
            
            if radius_scaling is not None:
                radius*=radius_scaling
                
            info['pos1']=pos1
            info['pos2']=pos2
            info['radius']=radius
            
            vec=pos2-pos1
            vec_amp=np.sqrt(np.sum(vec**2, axis=1))# norm
            vec_amp[vec_amp==0]=1.0 # avoid divide by zero
            vec_norm=vec/vec_amp[:, None]
            
            # for edges of length < 2 set to length to 3 to avoid diconnedted maps
            vec_amp[vec_amp<2.0]=2.0

            
            info['vec_amp']=vec_amp
            info['vec_norm']=vec_norm
            
            
        if self.tags['so2']:
            
            so21=np.array([g.node[i[0]]['so2'] for i in e])
            so22=np.array([g.node[i[1]]['so2'] for i in e]) 
            info['so21']=so21
            info['so22']=so22     
        
        if self.tags['hct']:
            types=np.array([g.node[i[0]]['type'] for i in e])
            if types.max()==3: types-=1 # types should be 0-->Art., 1-->Vein, 2-->Capp
            info['types']=types    
        
        if self.tags['velocity']:
        
            velocity=np.array([g.node[i[0]]['velocity'] for i in e])
            dx=np.array([g.node[i[0]]['dx'] for i in e])
            dy=np.array([g.node[i[0]]['dy'] for i in e])
            dz=np.array([g.node[i[0]]['dz'] for i in e])
            
            info['velocity']=velocity
            info['dx']=dx
            info['dy']=dy
            info['dz']=dz
        
        
        if self.tags['segments']:
            
            b=[g.node[i[0]]['label'] for i in e]
            info['segments']=b
        
        
        if self.tags['propagation']:
            # this is for mapping propagation/branching level
            try:
                label1=np.array([g.node[i[0]]['branch'] for i in e])
                label2=np.array([g.node[i[1]]['branch'] for i in e])
            
                info['label1']=label1
                info['label2']=label2   
            except:
                print('--Cannot return \'propagation\'; no labels on input graph!')
                self.tags['propagation']=False
            
        return info
    

    def GetOutput(self, 
                 radius_scaling=None,
                 hct_values=[0.33, 0.44, 0.44]):
        '''
        
        Input:
            resolution: This in the number of points interplated at each graph edge
            radius_scaling: This factor used to increase/decrease the overll radius size
            hct_values: A list in the format [hct_in_arteriols, hct_in_venules, hct_in_cappilaries]        
        '''
        
        info = self.get_cylinder_infos(self.g, radius_scaling=radius_scaling)
        real_s, new_s, niter = self.real_s, self.new_s, self.niter

        if self.tags['binary']:
            binary_image=np.zeros(new_s) 
        
        if self.tags['so2']:
            so2_image = np.zeros(new_s) 
        
        if self.tags['hct']:
            hct_values=np.array(hct_values)
            info['hct']=hct_values[info['types']]
            hct_image = np.zeros(new_s) 
        
        if self.tags['velocity']:
            vx_image = np.zeros(new_s) 
            vy_image = np.zeros(new_s) 
            vz_image = np.zeros(new_s) 
            vel_image = np.zeros(new_s)  # velocity
        
        if self.tags['gradient']:
            grad_image = np.zeros(new_s)  # gradient image to capture propagation across graph
        
        if self.tags['segments']:
            segments_image = np.zeros(new_s)  #  image to capture clustered segments        
            
        for idx in tqdm(range(niter)):
            
            if self.tags['binary']:
                p1, p2, r = info['pos1'][idx], info['pos2'][idx], info['radius'][idx]            
            
                # direction of this segment
                vec_amp, vec_norm = info['vec_amp'][idx], info['vec_norm'][idx]
                
            if self.tags['so2']:
                s1, s2 = info['so21'][idx], info['so22'][idx]
            else:
                s1, s2 = 0, 0
            
            if self.tags['hct']:
                h = info['hct'][idx]
            else:
                h = 0
                
            if self.tags['velocity']:
                velo = info['velocity'][idx]
            else:
                velo = 0
               
            if self.tags['propagation']:
                l1, l2 = info['label1'], info['label2']
            else:
                l1, l2 = 0, 0
        
            if self.tags['segments']:
                branch = info['segments'][idx][0]
            else:
                branch=0


            c, shape  = self.cylinder(vec_norm, radius=r, length=vec_amp)
            
            x0, y0, z0 = c[0], c[1], c[2]       
                                
            pos=(p1+p2)/2.0
            
            # this to align to middle of the cylinder
            sub=np.array(shape)/2.0
            x = x0-sub[0]
            y = y0-sub[1]
            z = z0-sub[2]
            
            # this to align in the middle of the edge
            x=x+pos[0]
            y=y+pos[1]
            z=z+pos[2]
            
            ss=s1
            ll=l1
            
            c=(x.astype(int), y.astype(int), z.astype(int))
            
            if np.size(np.array(c))==0:
                print('Zero length edge: '+str(idx))
                continue

                
            if self.tags['binary']:
                binary_image[c]=1
            
            if self.tags['so2']:
                so2_image[c]=ss
            
            if self.tags['hct']:
                hct_image[c]=h
                
            if self.tags['segments']:
                segments_image[c]=branch
                
            if self.tags['velocity']:
                vx_image[c]=velo*vec_norm[0]
                vy_image[c]=velo*vec_norm[1]
                vz_image[c]=velo*vec_norm[2]
                vel_image[c]=velo
            
            if self.tags['gradient']:
                grad_image[c]=ll
    
    
        x0,x1=self.ends[0]
        y0,y1=self.ends[1]
        z0,z1=self.ends[2]
        
        ret=dict()
        # cropping to recover origional image size
        if self.tags['binary']:
            binary_image=binary_image[x0:x1, 
                                      y0:y1, 
                                      z0:z1]
            ret['binary'] = binary_image.astype(int)
            
        if self.tags['so2']:
            so2_image=so2_image[x0:x1, 
                                y0:y1, 
                                z0:z1]
            ret['so2'] = so2_image.astype('float32')
            
        if self.tags['hct']:
            hct_image=hct_image[x0:x1, 
                                y0:y1, 
                                z0:z1]
            ret['hct'] = hct_image.astype('float32')
        
        if self.tags['velocity']:
            
            vx_image=vx_image[x0:x1, 
                              y0:y1, 
                              z0:z1]
            ret['vx'] = vx_image.astype('float32')
            vy_image=vy_image[x0:x1, 
                              y0:y1, 
                              z0:z1]
            ret['vy'] = vy_image.astype('float32')
            vz_image=vz_image[x0:x1, 
                              y0:y1, 
                              z0:z1]
            ret['vz'] = vz_image.astype('float32')
            vel_image=vel_image[x0:x1, 
                                y0:y1, 
                                z0:z1]
            ret['velocity'] = vel_image.astype('float32')
            
        if self.tags['gradient']:
            grad_image=grad_image[x0:x1, 
                                  y0:y1, 
                                  z0:z1]         
            ret['gradient']= grad_image.astype('float32')
            
            
        if self.tags['segments']:
            segments_image=segments_image[x0:x1, 
                                          y0:y1, 
                                          z0:z1]
            ret['segments']= segments_image.astype('float32')            
            
        return ret

def normalize(im):
    return ((im-im.min())/(im.max()-im.min()))
 
def save_3DRGB(name, img, dtype='uint16'):
    
    if dtype=='uint16':
        c=2**16
    elif dtype=='uint8':
        c=2**8
    else:
        c=1
    
    # coloring segments
    img = img/img.max()
    cmap = plt.get_cmap('rainbow')
    sh=img.shape
    ## MODIFIED BY LG 2020_06_14
    #img = cmap(img.ravel()).reshape([sh[0],sh[0],sh[0],4])
    img = cmap(img.ravel()).reshape([sh[0],sh[1],sh[2],4])
    ## END OF MODIFICATION BY LG 2020_06_14

    
    img = np.delete(img, 3, 3)
    img=(img*c).astype('uint16')
    skio.imsave(name, img)
    
    
def save_GRAY(name, im, dtype='uint16'):
    
    if dtype=='uint16':
        c=2**16
    elif dtype=='uint8':
        c=2**8
    else:
        c=1
    
    # coloring segments
    im=((im-im.min())/(im.max()-im.min()))*c
    im=im.astype(dtype)
    skio.imsave(name, im)    

## ADDED BY LG 2020_06_10:
def keep_single_segment(im3d, segment_index):
        #segment_index = 3
        #single_segment_mask = np.zeros(im3d.shape, dtype = uint8)
        unique_values = np.unique(im3d)


        single_segment_mask = (im3d == unique_values[segment_index])
        #single_segment_mask[np.argwhere(im3d == unique_values[segment_index])] = 255
        return single_segment_mask

#single_segment = keep_single_segment(segments, 10)

#skio.imsave('segment_white.tif', single_segment)   

## END OF ADDITION BY LG 2020_06_10:

    
if __name__=='__main__':

    ## START OF ADDITION BY LG 2020_11_08:
    radius_scaling_factor = 0.4
    ## END OF ADDITION BY LG 2020_11_08:
    '''
    # binary image
    #segpath='D:\code\VascularGraph\synth1.mat'
    #segpath = 'binary_mask.mat'
    #segpath = 'D:\Downloads\TLP1_Thy2_GCaMP7_FITC_mouse_930nm_1024l_1slow_3p5x_zoom_+100um_heigh_summed_bc_t_FOV1_062_summed_bc_top180_slices_rs512x2048_for_vg_gamma2_rs_512x512x180_clij2_percentile_binary_3d_median_2x2x2.mat'
    '''
    segpath = os.path.join('D:\Downloads', 'binary_mask_100x130x180.mat')
    pajek_fn = segpath.replace('.mat','.pajek')
    
    seg=vg.GraphIO.ReadStackMat(segpath).GetOutput()
    
    # graph
    ## MODIFIED BY LG 2020_06_14
    #g=vg.Skeletonize.Skeleton(seg).Update(ret=True)
    #g = vg.GraphIO.ReadPajek('demo_graph_synth1.pajek').GetOutput()
    #g = vg.GraphIO.ReadPajek('D:\Downloads\mygraph_2020_01_13_512x512x180.pajek').GetOutput()
    g = vg.GraphIO.ReadPajek(pajek_fn).GetOutput()
    ## END OF MODIFICATION BY LG 2020_06_14



    #give each segment a unique id, bifurcation nodes hold multiple ids
    g.LabelSegments()
    
    #fixed radius across each segment (optional)
    #g.RefineRadiusOnSegments(rad_mode='max')  # commented out by LG 2020_08_24
    
    # reconstructed image from piecewise cylinderical models (voxels of each segment will have a unique intensity value)
    
    # if a similar dimension is needed
    ret=CreateCylinderMappings(g, im_size=seg.shape, to_return=['segments']).GetOutput(radius_scaling = radius_scaling_factor)
    
    # # if dimension is not necessary need to match that of the original segmented image
    #ret=CreateCylinderMappings(g, to_return=['segments']).GetOutput() 

    
    segments=ret['segments']

    ## MODIFIED BY LG 2020_06_19
    print('shape of segments is ' + str(segments.shape))
    print(f'max of segments is {np.max(segments)}')
    print(f'max of segments is {np.min(segments)}')
    print(f'number of unique values of segments is {np.size(np.unique(segments))}')
    
    #print(segments)

    ## END OF MODIFICATION BY LG 2020_06_14




    # save Gray
    save_GRAY('segments_gr.tif', segments)


    
    # save RGB
    save_3DRGB('segments_rgb.tif', segments)
    
    
    
    
    