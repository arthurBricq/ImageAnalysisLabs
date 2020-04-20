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

transform = skimage.transform.SimilarityTransform(scale=0.9)
img1 = ones[1]
img2 = skimage.transform.warp(img1, transform)

# Then compute the fourier transform
[X1, Y1] = imPro.get_outmost_contour(img1)
[X2, Y2] = imPro.get_outmost_contour(img2)
fourier1 = np.fft.fft(X1 + 1j * Y1)
fourier2 = np.fft.fft(X2 + 1j * Y2)
# fourier1 = normalize_fourier(fourier1)
# fourier2 = normalize_fourier(fourier2)
fourier1 = np.fft.fftshift(fourier1)
fourier2 = np.fft.fftshift(fourier2)

imPro.plot_fourier_descriptors(fourier1)
imPro.plot_fourier_descriptors(fourier2)

# Plot the 2 images just to see if it is correct
fig, axs = plt.subplots(1, 2, figsize=(5, 5))
axs[0].imshow(img1)
axs[1].imshow(img2)

#
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


# %% Rotation invariance

img = ones[1]
theta = np.pi/4
R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
X1, Y1 = imPro.get_outmost_contour(img)
contour1 = np.column_stack([X1, Y1])
contour2 = contour1 @ R
X2, Y2 = contour2[:, 0], contour2[:, 1]

# Fourier
signal1 = X1 + 1j * Y1
fourier1 = np.fft.fft(signal1)
signal2 = X2 + 1j * Y2
fourier2 = np.fft.fft(signal2)

# Fourier Invariant
fourier1_n = imPro.rotation_invariance(fourier1)
fourier2_n = imPro.rotation_invariance(fourier2)

# Plotting all this 
fig, axs = plt.subplots(1,2,figsize=(10,5))
axs[0].plot(np.angle(fourier1_n), label = 'fourier')
axs[0].plot(np.angle(fourier2_n),'-.', label = 'invariant fourier')
axs[1].plot(np.abs(fourier1_n), label = 'fourier')
axs[1].plot(np.abs(fourier2_n),'-.', label = 'invariant fourier')
axs[0].legend()
axs[1].legend()

# %% 


