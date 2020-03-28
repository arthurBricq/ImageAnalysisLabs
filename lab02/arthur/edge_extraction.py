# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:14:26 2020

@author: abric
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")

import lab02_functions as imPro
import skimage, skimage.feature, skimage.morphology

#%% Get the data to use

zeros = imPro.get_zeros()
ones = imPro.get_ones()

#%% Try to extract edge for one picture
im = ones[0]

im = skimage.morphology.area_closing(im,area_threshold=100)
edgeMap = skimage.feature.canny(im)
[X,Y] = np.where(edgeMap)
fig, ax = plt.subplots(1, 1, figsize=(10,10))
ax.imshow(edgeMap,cmap='gray')

#%% Function to extract the edge 

def get_outmost_contour(img):
    """
    Image analysis lab 2
    Returns the contour (outside contour) of the number (in grayscale) given as parameter
    
    Parameters
    -----------
    image : grayscale image of a number
    
    Returns 
    -------
    contour : [X,Y] np.array of all the pixels in the contour
    
    """
    tmp = skimage.morphology.area_closing(img,area_threshold=100)
    edgeMap = skimage.feature.canny(tmp)
    [X,Y] = np.where(edgeMap)
    return X,Y

get_outmost_contour()

#%% Make it for all points 

fig, axes = plt.subplots(2, len(zeros), figsize=(12, 3))

for i,ax in enumerate(axes[0]):
    ax.imshow(zeros[i],cmap = 'gray')
    ax.axis('off')
for i,ax in enumerate(axes[1]):
    im = zeros[i]
    im = skimage.morphology.area_closing(im,area_threshold=250)
    edgeMap = skimage.feature.canny(im)
    ax.imshow(edgeMap,cmap = 'gray')
    ax.axis('off')
    
fig, axes = plt.subplots(2, len(ones), figsize=(12, 3))
for i,ax in enumerate(axes[0]):
    ax.imshow(ones[i],cmap = 'gray')
    ax.axis('off')
for i,ax in enumerate(axes[1]):
    im = ones[i]
    im = skimage.morphology.area_closing(im,area_threshold=250)
    edgeMap = skimage.feature.canny(im)
    ax.imshow(edgeMap,cmap = 'gray')
    ax.axis('off')


