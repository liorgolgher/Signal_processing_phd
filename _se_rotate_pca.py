# conda activate cv
import porespy as ps
import os
import gdal
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage.filters import threshold_otsu
from skimage.filters import try_all_threshold
import cv2
import math
from skimage import img_as_ubyte
from skimage.transform import radon
from skimage.morphology import convex_hull_image
import pandas as pd  
import time
import tifffile as tf

for filenum in np.arange(12):
    filename = 'D:/Downloads/v/' + str(1+filenum) + '.tif'
    #file='D:/Downloads/v/1.tif'

    arr = tf.imread(filename)
    
    '''
    a = ps.generators.cylinders((200,200,20), 10, ncylinders=1, phi_max=0, length=100)
    a = 1-a
    a = np.sum(a,axis=2)
    a = a * (0.3 + np.random.random(size=(a.shape)))
    arr = a.copy()
    '''
    #
    (cols, rows) = arr.shape
    thresh = threshold_otsu(arr)
    binary = arr > thresh
    plt.imshow(binary)
    plt.show(block='False')
    plt.pause(1)
    plt.close('all')
    #radon_sinogram = radon(image, theta=theta, circle=True)
    '''
    radon_sinogram = radon(binary)
    plt.imshow(radon_sinogram)
    plt.show()
    '''


    points = binary>0
    y,x = np.nonzero(points)
    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.vstack([x, y])
    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]
    evec1, evec2 = evecs[:, sort_indices]
    x_v1, y_v1 = evec1  
    x_v2, y_v2 = evec2
    print(evec1)
    print(evec2)
    
    scale = 40
    plt.plot([x_v1*-scale*2, x_v1*scale*2],
            [y_v1*-scale*2, y_v1*scale*2], color='red')
    plt.plot([x_v2*-scale, x_v2*scale],
            [y_v2*-scale, y_v2*scale], color='blue')
    plt.plot(x,y, 'y.')
    plt.axis('equal')
    plt.gca().invert_yaxis()  
    plt.show(block='False')
    plt.pause(1)
    plt.close('all')
    vascular_angle_degrees = np.tanh((x_v1)/(y_v1))  * 180 /(math.pi)
    print('theta = ' + str(vascular_angle_degrees))
    radon_vals = radon(arr, circle=False, theta=(vascular_angle_degrees, vascular_angle_degrees+90) )
    print(radon_vals.shape)
    normed_radon_vals = radon_vals - np.min(radon_vals,axis=0,keepdims=True)
    normed_radon_vals = np.divide(normed_radon_vals,    np.max(normed_radon_vals,axis=0,keepdims=True))
    print(normed_radon_vals)