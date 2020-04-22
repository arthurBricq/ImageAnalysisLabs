# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:59:03 2020

@author: abric
"""

import importlib
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")


import lab02_functions as imPro
from imPro import PlotData
import skimage
import skimage.transform
from numpy.fft import fft

importlib.reload(imPro)

# %% Get the image and the contour

zeros = imPro.get_zeros()
ones = imPro.get_ones()


# %% Normalize routine testing 1 

# https://dsp.stackexchange.com/questions/19982/fourier-descriptors-trying-to-classify-objects
# http://fourier.eng.hmc.edu/e161/lectures/fd/node1.html

transform = skimage.transform.SimilarityTransform(rotation=0.2)
img1 = ones[1]
img2 = skimage.transform.warp(img1, transform)

# Then compute the fourier transform
[X1, Y1] = imPro.get_outmost_contour(img1)
[X2, Y2] = imPro.get_outmost_contour(img2)
fourier1 = np.fft.fft(X1 + 1j * Y1)
fourier2 = np.fft.fft(X2 + 1j * Y2)
# fourier1 = imPro.rotation_invariance(fourier1)
fourier2 = imPro.rotation_invariance(fourier2)


fourier1 = np.fft.fftshift(fourier1)
fourier2 = np.fft.fftshift(fourier2)
imPro.plot_fourier_descriptors(fourier1)
imPro.plot_fourier_descriptors(fourier2)

# Plot the 2 images just to see if it is correct
fig, axs = plt.subplots(1, 2, figsize=(5, 5))
axs[0].imshow(img1)
axs[1].imshow(img2)


# %% Rotation effect (the modulus is constant)
    
fig, axs = plt.subplots(1,3,figsize = (15,5))
img = ones[1]
imPro.plot_FD_rotation_invariance(img, np.pi/16, axs[0])
imPro.plot_FD_rotation_invariance(img, np.pi/8, axs[1])
imPro.plot_FD_rotation_invariance(img, np.pi/4, axs[2])
axs[0].set_ylabel('Phase [rad]')
fig.suptitle("Effect of rotation over the phase of the Fourier Descriptors")

# %% Scalling effect (the phase is constant)

fig, axs = plt.subplots(1,3,figsize = (15,5))
img = ones[1]
imPro.plot_FD_scaling_invariance(img, 1, axs[0])
imPro.plot_FD_scaling_invariance(img, 1.2, axs[1])
imPro.plot_FD_scaling_invariance(img, 1.5, axs[2])
axs[0].set_ylabel('FD Amplitudes')
fig.suptitle("Effect of scalling over the phase of the Fourier Descriptors")


# %% Invariance Assessement

### Transformation parameters
# 1. rotation
theta = np.pi/4 * 0 
R_rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
# 2. scalling
a = 1.3
R_sca = np.array([[a,0],[0,a]])
# 3. translation 
tx = 2
ty = 0
# 4. Starting point 
k_0 = 1

### Construct the fourier descriptors
img = ones[1]
X1, Y1 = imPro.get_outmost_contour(img)
contour1 = np.column_stack([X1, Y1])
R = R_sca @ R_rot
contour2 = contour1 @ R
contour2 = contour2 + tx + 1j*ty
X2, Y2 = contour2[:, 0], contour2[:, 1]
if k_0 > 0:
    X2 = np.concatenate((X2[k_0:],X2[0:k_0]))
    Y2 = np.concatenate((Y2[k_0:],Y2[0:k_0]))
    
signal1 = X1 + 1j * Y1
fourier1 = np.fft.fft(signal1)
signal2 = X2 + 1j * Y2
fourier2 = np.fft.fft(signal2)

### Make fourier Invariant
fourier1_n = imPro.translation_invariance(fourier1)
fourier2_n = imPro.translation_invariance(fourier2)
fourier1_n = imPro.scaling_invariance(fourier1)
fourier2_n = imPro.scaling_invariance(fourier2)
fourier1_n = imPro.starting_point_invariance(fourier1_n)
fourier2_n = imPro.starting_point_invariance(fourier2_n)

# Plotting all this 
fig, axs = plt.subplots(1,2,figsize=(10,5))
axs[0].plot(np.angle(fourier1_n), label = 'fourier')
axs[0].plot(np.angle(fourier2_n),'-.', label = 'invariant fourier')
axs[1].plot(np.abs(fourier1_n), label = 'fourier')
axs[1].plot(np.abs(fourier2_n),'-.', label = 'invariant fourier')
axs[0].legend()
axs[1].legend()


# %% Trying some robust classification 



#%% Plotting results 
    
# =============================================================================
# We want to make a plot illustrating several cases, and some of them including different feature vectors. 
# =============================================================================

    
    
# 1. Construction of the experiment (data-model to generate plot easily)
plots_to_make = [] 
plots_to_make.append(PlotData("Not invariant", [0,0,0,0], [0,1]))
plots_to_make.append(PlotData("Not invariant", [0,0,0,0], [5,6]))
plots_to_make.append(PlotData("Rotation invariant", [1,0,0,0], [0, 1]))
plots_to_make.append(PlotData("Starting-Point invariant", [0,0,0,1], [3, 0]))
plots_to_make.append(PlotData("Translation + Starting-Point + Scalling", [0,1,1,1], [3, 2]))
plots_to_make.append(PlotData("Starting-Point invariant", [0,0,0,1], [3, 2]))
plots_to_make.append(PlotData("Translation + Starting-Point + Scalling", [0,1,1,1], [3, 4]))


fig, axs = plt.subplots(2,4,figsize = (25,10))
axs = axs.ravel()
for i, plotData in enumerate(plots_to_make):
    x_zeros = []
    x_ones = []
    ax = axs[i]
    for img in zeros: x_zeros.append(imPro.get_feature_vector(img, plotData.invariances))
    for img in ones: x_ones.append(imPro.get_feature_vector(img, plotData.invariances))
    x_zeros = np.array(x_zeros)
    x_ones = np.array(x_ones)
    ax.plot(x_zeros[:,plotData.features[0]], x_zeros[:,plotData.features[1]],'.b', label = 'zeros')
    ax.plot(x_ones[:,plotData.features[0]], x_ones[:,plotData.features[1]],'.r', label = 'ones')
    labels = plotData.get_features_label()
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(plotData.name)
    ax.legend()
fig.subplots_adjust(hspace=0.4)
fig.delaxes(axs[-1])
fig.suptitle("Projection of features, for different images and different normalization techniques")
    
    
    

#%% Code to construct one feature vector

# Feature vectors construction
x_zeros_ = []
x_ones = []
for img in zeros: x_zeros.append(get_feature_vector(img))
for img in ones: x_ones.append(get_feature_vector(img))
x_zeros = np.array(x_zeros)
x_ones = np.array(x_ones)

# Plot the results
plt.figure()
fig, axs = plt.subplots(2,1, figsize = (8,12))

# Select the features to use (within the feature vector) 
x0, x1, x2, x3 = 0, 1, 2, 3
axs[0].plot(x_zeros[:,x0], x_zeros[:,x1],'.b', label = 'zeros')
axs[0].plot(x_ones[:,x0], x_ones[:,x1],'.r', label = 'ones')
axs[0].set_xlabel('x0')
axs[0].set_ylabel('x1')
axs[0].legend()

axs[1].plot(x_zeros[:,x2], x_zeros[:,x3],'.b', label = 'zeros')
axs[1].plot(x_ones[:,x2], x_ones[:,x3],'.r', label = 'ones')
axs[1].set_xlabel('x2')
axs[1].set_ylabel('x3')
axs[1].legend()

plt.show()