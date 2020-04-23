#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:43:41 2020

@author: arthur
"""

import importlib
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")


import lab02_functions as imPro
from lab02_functions import PlotData
import skimage
import skimage.transform
from numpy.fft import fft

importlib.reload(imPro)


#%% Load the data 

zeros = imPro.load_img_seq(file_path='lab-02-data/part1/0')
ones = imPro.load_img_seq(file_path='lab-02-data/part1/1')
twos = imPro.load_img_seq(file_path='lab-02-data/part2/2')
threes = imPro.load_img_seq(file_path='lab-02-data/part2/3')

#%% Just to make sure every thing works fine 

imgs = [zeros[0], ones[1], twos[2], threes[0]]
fig, axs = plt.subplots(1,4,figsize=(10,5))
for img, ax in zip(imgs, axs):
    contour = imPro.get_outmost_contour(img)
    img_c = imPro.get_contour_image(contour)
    ax.imshow(img_c)
fig.suptitle('Extracted contours for 4 different images of the dataset')


#%% FD

# 1. Construction of the experiment (make the data-model to generate plot easily)
plots_to_make = [] 
plots_to_make.append(PlotData("Not invariant", [0,0,0,0], [0,1]))
plots_to_make.append(PlotData("Not invariant", [0,0,0,0], [5,6]))
plots_to_make.append(PlotData("Rotation invariant", [1,0,0,0], [0, 1]))
plots_to_make.append(PlotData("Starting-Point invariant", [0,0,0,1], [3, 0]))
plots_to_make.append(PlotData("Starting-Point invariant", [0,0,0,1], [3, 2]))
plots_to_make.append(PlotData("Translation + Starting-Point + Scalling", [0,1,1,1], [3, 2]))
plots_to_make.append(PlotData("Translation + Starting-Point + Scalling", [0,1,1,1], [3, 4]))

# 2. Make some nice plots illustrating all this
fig, axs = plt.subplots(3,3,figsize = (15,12))
axs = axs.ravel()
for i, plotData in enumerate(plots_to_make):
    x_zeros, x_ones, x_twos, x_threes = [], [], [], []
    ax = axs[i]
    for img in zeros: x_zeros.append(imPro.get_feature_vector(img, plotData.invariances))
    for img in ones: x_ones.append(imPro.get_feature_vector(img, plotData.invariances))
    for img in twos: x_twos.append(imPro.get_feature_vector(img, plotData.invariances))
    for img in threes: x_threes.append(imPro.get_feature_vector(img, plotData.invariances))
    x_zeros, x_ones, x_twos, x_threes = np.array(x_zeros), np.array(x_ones), np.array(x_twos), np.array(x_threes)
    ax.plot(x_zeros[:,plotData.features[0]], x_zeros[:,plotData.features[1]],'.b', label = 'zeros')
    ax.plot(x_ones[:,plotData.features[0]], x_ones[:,plotData.features[1]],'.r', label = 'ones')
    ax.plot(x_twos[:,plotData.features[0]], x_twos[:,plotData.features[1]],'.y', label = 'twos')
    ax.plot(x_threes[:,plotData.features[0]], x_threes[:,plotData.features[1]],'.g', label = 'threes')
    labels = plotData.get_features_label()
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title("({}) ".format(i+1) + plotData.name)
    ax.legend()
fig.subplots_adjust(hspace=0.4, wspace = 0.3)
fig.delaxes(axs[-1]), fig.delaxes(axs[-2])
fig.suptitle("Projection of features, for different images and different normalization techniques");

#%% AA's function 

def region_based_features(images):
    """
    Returns an array of region based features for all the images in the input
    
    :param images:  group of grayscale images of numbers
    :return: param_images np.array of all the region based features of the input images
    """
    param_images = []
    for img in images:
        binary=imPro.binarize(img)
        param=[]
        param.append(imPro.area(binary))
        param.append(imPro.perimeter(img))
        param.append(imPro.compacity(img))
        inertias = imPro.inertia(img)
        param.append(inertias[0])
        param.append(inertias[1])
        param_images.append(param)
    param_images = np.array(param_images)
    return param_images

#%% Region based

param_zeros = region_based_features(zeros)
param_ones = region_based_features(ones)
param_twos = region_based_features(twos)
param_threes = region_based_features(threes)


fig, axs = plt.subplots(1,4, figsize = (16,4))
axs[0].plot(param_zeros[:,0], param_zeros[:,1],'.b', label = 'zeros')
axs[0].plot(param_ones[:,0], param_ones[:,1],'.r', label = 'ones')
axs[0].plot(param_twos[:,0], param_twos[:,1],'.y', label = 'twos')
axs[0].plot(param_threes[:,0], param_threes[:,1],'.g', label = 'threes')
axs[0].set_xlabel('Area ')
axs[0].set_ylabel('Perimeter ')
axs[0].set_title('Region based descriptors')
axs[0].legend()

axs[1].plot(param_zeros[:,0], param_zeros[:,2],'.b', label = 'zeros')
axs[1].plot(param_ones[:,0], param_ones[:,2],'.r', label = 'ones')
axs[1].plot(param_twos[:,0], param_twos[:,2],'.y', label = 'twos')
axs[1].plot(param_threes[:,0], param_threes[:,2],'.g', label = 'threes')
axs[1].set_xlabel('Area ')
axs[1].set_ylabel('Compacity ')
axs[1].set_title('Region based descriptors')

axs[2].plot(param_zeros[:,1], param_zeros[:,2],'.b', label = 'zeros')
axs[2].plot(param_ones[:,1], param_ones[:,2],'.r', label = 'ones')
axs[2].plot(param_twos[:,1], param_twos[:,2],'.y', label = 'twos')
axs[2].plot(param_threes[:,1], param_threes[:,2],'.g', label = 'threes')
axs[2].set_xlabel('Perimeter ')
axs[2].set_ylabel('Compacity ')
axs[2].set_title('Region based descriptors')

axs[3].plot(param_zeros[:,3], param_zeros[:,4],'.b', label = 'zeros')
axs[3].plot(param_ones[:,3], param_ones[:,4],'.r', label = 'ones')
axs[3].plot(param_twos[:,3], param_twos[:,4],'.y', label = 'twos')
axs[3].plot(param_threes[:,3], param_threes[:,4],'.g', label = 'threes')
axs[3].set_xlabel('Principal inertia ')
axs[3].set_ylabel('Second inertia ')
axs[3].set_title('Region based descriptors')


















