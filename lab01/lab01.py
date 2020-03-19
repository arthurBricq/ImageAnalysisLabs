# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:27:39 2020

@author: Arthur Bricq
"""

import matplotlib.pyplot as plt
import lab01_functions as imPro
import skimage.io
import skimage.morphology
from skimage.color import rgb2gray
import numpy as np
import YonasModule.morphological_edge_detection as IP_med


#%% PART I *******************************

# Load image
brain_im = skimage.io.imread('data/lab-01-data/brain-slice40.tiff')
im_h, im_w = brain_im.shape

# Plot the image
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.imshow(brain_im, cmap='gray')
ax.set_title('MRI brain image ({} px, {} px)'.format(im_h, im_w))
ax.axis('off')
plt.show()

#%% Functions to encapsulate the 2 edge detectors that were done

# Those 2 functions return an edge map with negative pixels where the edges are
# Those edge maps are then used in a labelization algortimh to either
# - count the pixels (part I)
# - cluster the pixels in groups and find the number of different shapes


def getEdgeMapMethod1(img, t1, t2):
    binary_im = np.logical_and(img > t1, brain_im < t2) 
    edgeMap = skimage.filters.laplace(binary_im)
    edgeMap[edgeMap>0]=-edgeMap[edgeMap>0]
    return edgeMap
    
def getEdgeMapMethod2(img, t1, t2):
    sigma=2.5
    G_t1=10
    G_t2=200
    (edges,thld_img,G,Phi,G_thld_img)=imPro.skeletonize_based_sobel(img,sigma,t1,t2,G_t1,G_t2)
    edgeMap = -np.array(edges).astype(int)
    return edgeMap
    

#%% Edge detection 1 (arthur) or 2 (Jonas)
    
# Best thresholds for arthur: 40 to 100
# Best thresholds for jonas: 55 to 105  (?)

# Example on how to use the edge detection function 
edgeMap = getEdgeMapMethod2(brain_im,40,100)
imPro.plotEdgeMap(edgeMap)

#%% labelization 

# edgeMap = getEdgeMapMethod1(brain_im,40,100)
edgeMap = getEdgeMapMethod2(brain_im,55,105)
imPro.plotEdgeMap(edgeMap)

# Get the labels to count the number of pixels
labels = imPro.labelizePixels(edgeMap)
imPro.plotImage(labels)

# Analysis of the labels
plt.figure()
# 1. Histogram of each label
n, bins, patches = plt.hist(labels.ravel(),bins=np.unique(labels))
# Sort the occurences in ascending order
indexes = np.argsort(n)
# The label of interest is clearly the second most recurent label
indexOfLabel = indexes[-2]
numberOfPixels = n[indexOfLabel] 
ratio = numberOfPixels / im_h / im_w
print("Result of the analysis")
print("- Number of pixels: {},\n- Ratio: {}".format(numberOfPixels, ratio))


#%% PART II **************************

im1 = skimage.io.imread('data/lab-01-data/arena-shapes-01.png')
im2 = skimage.io.imread('data/lab-01-data/arena-shapes-02.png')
im3 = skimage.io.imread('data/lab-01-data/arena-shapes-03.png')
images = np.array([im1,im2,im3])


#%% 

im = im2
imPro.plotImage(im)


# 1. Seperate the background from the shapes with optimal threshold 
tmp = im[:,:,0]
imPro.plotImage(tmp)
t = skimage.filters.threshold_otsu(tmp)
im_bi = tmp < t 
imPro.plotImage(im_bi)

# 2. Obtain the black shapes
tmp = im[:,:,2]
tmp = skimage.filters.gaussian(tmp,sigma = 2.5) * 255
tmp = skimage.exposure.rescale_intensity(tmp,out_range = (0,255))
im_bi_black = imPro.blueThresholdForPixel((358,56),tmp)
imPro.plotImage(im_bi_black)
plt.title('Black parts of the image')

# Obtain the blue shapes
im_bi_blue = np.logical_xor(im_bi,im_bi_black)
im_bi_blue = skimage.morphology.opening(im_bi_blue)
imPro.plotImage(im_bi_blue)
plt.title('Blue parts of the image')

# Count the number of pixels and of each shape, and dislay the result
print("COULEUR BLEUE")
groups_blue = imPro.countNumberOfShapes(im_bi_blue)
print("COULEUR NOIRE")
groups_black = imPro.countNumberOfShapes(im_bi_black)






