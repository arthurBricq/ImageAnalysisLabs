# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:27:39 2020

@author: Arthur Bricq
"""

import matplotlib.pyplot as plt
import lab01_functions as imPro
import skimage.io
import skimage.morphology
import numpy as np
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.segmentation import chan_vese
from skimage.feature import canny
from scipy import ndimage
from skimage.color import rgb2gray
from skimage import exposure


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

def getEdgeMapMethod3(img):
    edgeMap = canny(img, sigma= 1.1)
    edgeMap = skimage.morphology.dilation(edgeMap)
    edgeMap = -edgeMap.astype(int)
    imPro.plotEdgeMap(edgeMap)
    return edgeMap
 

#%% labelization 

# edgeMap = getEdgeMapMethod1(brain_im,40,100)
edgeMap = 3*getEdgeMapMethod3(brain_im)

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

#%%
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

#%% To count the number of pixels on each image


for i,im in enumerate(images):
    # 1. Seperate the background from the shapes with optimal threshold on the red channel
    tmp = im[:,:,0]
    t = skimage.filters.threshold_otsu(tmp)
    im_bi = tmp < t 

    # 2. Obtain the black shapes
    tmp = im[:,:,2]
    tmp = skimage.filters.gaussian(tmp,sigma = 2.5) * 255
    tmp = skimage.exposure.rescale_intensity(tmp,out_range = (0,255))
    im_bi_black = imPro.blueThresholdForPixel((358,56),tmp)
    
    # 3. Obtain the blue shapes
    im_bi_blue = np.logical_xor(im_bi,im_bi_black)
    im_bi_blue = skimage.morphology.opening(im_bi_blue)

    # Count the number of pixels on each color
    n_blue = np.count_nonzero(im_bi_blue)
    n_black = np.count_nonzero(im_bi_black)
    
    print("Image :",i+1)
    print("Color blue: {} pixels".format(n_blue))
    print("Color black: {} pixels".format(n_black))
    print("\n")
    

    
#%% data processing function
    

    
#%% Picture by picture
im1 = skimage.io.imread('data/lab-01-data/arena-shapes-01.png')
im2 = skimage.io.imread('data/lab-01-data/arena-shapes-02.png')
im3 = skimage.io.imread('data/lab-01-data/arena-shapes-03.png')    

im = im2

# 1. Seperate the background from the shapes with optimal threshold on the red channel
tmp = im[:,:,0]
t = skimage.filters.threshold_otsu(tmp)
im_bi = tmp < t 
imPro.plotImage(im_bi)

# Get the edge map (to be used later) and create
edgeMap = skimage.filters.laplace(im_bi)
edgeMap[edgeMap>0]=-edgeMap[edgeMap>0]
imPro.plotEdgeMap(edgeMap)
labels = imPro.labelizePixels(edgeMap)

# 2. Obtain the black shapes
tmp = im[:,:,2]
tmp = skimage.filters.gaussian(tmp,sigma = 2.5) * 255
imPro.plotHistogram(tmp)
tmp = skimage.exposure.rescale_intensity(tmp,out_range = (0,255))
imPro.plotHistogram(tmp)
im_bi_black = imPro.blueThresholdForPixel((358,56),tmp)

# 3. Obtain the blue shapes
im_bi_blue = np.logical_xor(im_bi,im_bi_black)
im_bi_blue = skimage.morphology.opening(im_bi_blue)

# 5. Find the centers of each cluster.
centers_blue = findCentersOfClusters(im_bi_blue)
centers_black = findCentersOfClusters(im_bi_black)


# 6. From the clusters of each color, find the associated cluster in the edge map
# and count the number of pixels within those clusters.
pixels_blue = 0
for center in centers_blue.astype(int):
    label = labels[center[0],center[1]]
    count = labels[labels==label].shape[0]
    pixels_blue += count
    print(count)
print("BLEU {}".format(pixels_blue))

pixels_black = 0
for center in centers_black.astype(int):
    label = labels[center[0],center[1]]
    count = labels[labels==label].shape[0]
    pixels_black += count
    print(count)
print("BLACK {}".format(pixels_black))

# Count the number of pixels on each color
n_blue = np.count_nonzero(im_bi_blue)
n_black = np.count_nonzero(im_bi_black)

print("Color blue: {} pixels".format(n_blue))
print("Color black: {} pixels".format(n_black))
print("\n")





#%%

tmp = im1[:,:,0]
t = skimage.filters.threshold_otsu(tmp)
im_bi = tmp < t 
imPro.plotImage(im_bi)
edgeMap = skimage.filters.laplace(im_bi)
imPro.plotEdgeMap(edgeMap)



#%% Just trying some things

# Grayscale conv: black = 0 and white = 255

im = im3
im = rgb2gray(im)
background = im > 0.4
im[background] = 0.0
im = gaussian(im,sigma=0)
hist,bins = exposure.histogram(im)
im_bi = np.logical_and(im > 30.0 / 255, im < 70.0 / 255)

fig, axs = plt.subplots(1,3,figsize=(10,4))
axs[0].plot(hist[1:])
axs[1].imshow(im, cmap = 'gray')
axs[2].imshow(im_bi,cmap='gray')

#%%

# Grayscale conv: black = 0 and white = 255
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

im = im3
im = rgb2gray(im)
t = skimage.filters.threshold_otsu(im)
background = im > t 
im[background] = 0.0
im = gaussian(im,sigma=0)
hist,bins = exposure.histogram(im)
# the hist always has 2 peaks

array = savgol_filter(hist[1:],61,2)
peaks = find_peaks(array,height=50,distance=10)[0]

t1 = peaks[0] / 2 / 255
t2 = (peaks[1]-peaks[0]) / 255
im_bi = np.logical_and(im > t1 , im < t2)


fig, axs = plt.subplots(1,3,figsize=(10,4))
axs[0].plot(array)
axs[1].imshow(im > t1, cmap = 'gray')
axs[2].imshow(im_bi,cmap='gray')


#%% 

im_eq = exposure.equalize_hist(im)
imPro.plotImage(im_eq)

im_bi1 = im_eq > 0.04
im_bi2 = im < 0.4

fig, axs = plt.subplots(1,2,figsize=(10,4))
axs[0].imshow(im_bi2 > 0.04,cmap='gray')
axs[1].imshow(im_bi2,cmap='gray')

fig1, axs = plt.subplots(1,3,figsize=(10,4))
axs[0].imshow(im,cmap='gray')
axs[1].hist(im.ravel(), bins=256)
axs[2].hist(im_eq.ravel(), bins=256)
axs[1].set_yticklabels([])
axs[2].set_yticklabels([])



































