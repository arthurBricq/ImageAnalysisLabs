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

# %% Helper function 

def plot_fourier_descriptors(fourier):
    fix, axs = plt.subplots(2, 1, figsize=(5, 5))
    axs[0].plot(np.abs(fourier))
    axs[1].plot(np.angle(fourier))


# %% Normalize routine
# https://dsp.stackexchange.com/questions/19982/fourier-descriptors-trying-to-classify-objects

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

plot_fourier_descriptors(fourier1)
plot_fourier_descriptors(fourier2)
plt.show()

# Plot the 2 images just to see if it is correct
fig, axs = plt.subplots(1, 2, figsize=(5, 5))
axs[0].imshow(img1)
axs[1].imshow(img2)
plt.show()

# %% Rotation matrix (the modulus is invarient anyway)

theta = np.pi / 3
R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

img1 = ones[1]
X1, Y1 = imPro.get_outmost_contour(img1)
contour1 = np.column_stack([X1, Y1])
contour2 = contour1 @ R
X2, Y2 = contour2[:, 0], contour2[:, 1]

signal1 = X1 + 1j * Y1
fourier1 = np.fft.fft(signal1)

signal2 = X2 + 1j * Y2
fourier2 = np.fft.fft(signal2)

fig, ax = plt.subplots()
ax.plot(np.angle(fourier1), '-.r', label = 'contour')
ax.plot(np.angle(fourier2),'-.b', label = 'rotated contour')
ax.set_title('Effect of rotation over the Fourier Descriptors phase')
ax.legend()
plt.show()

# %% Scaling operation (the phase is invariant anyway)

R = np.array([[1.2, 0],
              [0, 1.2]])

img1 = ones[1]
X1, Y1 = imPro.get_outmost_contour(img1)
contour1 = np.column_stack([X1, Y1])
contour2 = contour1 @ R
X2, Y2 = contour2[:, 0], contour2[:, 1]

signal1 = X1 + 1j * Y1
fourier1 = np.fft.fft(signal1)

signal2 = X2 + 1j * Y2
fourier2 = np.fft.fft(signal2)

fig, ax = plt.subplots()
ax.plot(np.abs(fourier1)[1:5], '-.r', label = 'Contour')
ax.plot(np.abs(fourier2)[1:5],'-.b', label = 'Scaled contour')
ax.set_title('Effect of scaling over the Fourier Descriptors amplitude')
ax.set_xlabel('First harmonics')
ax.legend()
plt.show()


