# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:54:36 2020

@author: abricq
"""
import importlib
import matplotlib.pyplot as plt
import numpy as np 
import sys
sys.path.append("..")

import skimage.transform as transform
import skimage, skimage.feature, skimage.morphology
import lab02_functions as imPro
importlib.reload(imPro)
from numpy.fft import fft


#%% Example of FFT

# Check out this website: https://courspython.com/fft-introduction.html

n = 100
t = np.linspace(0,10,n)
signal = np.sin(2*3.14*t)

fourier = np.fft.fft(signal)
freqs = np.fft.fftfreq(n,10/n)
freqs, fourier = np.fft.fftshift(freqs), np.fft.fftshift(fourier)
amplitudes = np.abs(fourier)
phases = np.angle(fourier)

fig, axs = plt.subplots(2,1)
axs[0].plot(t,signal)
axs[1].plot(freqs, amplitudes,'-')

#%% Get the data to use
zeros=imPro.get_zeros()
ones = imPro.get_ones()

#%% Illustration of Fourier reconstruction with K harmonics
img = ones[1]
[X,Y] = imPro.get_outmost_contour(img)
im_contour = imPro.get_contour_image([X,Y])

signal = X + 1j * Y
fourier = np.fft.fft(signal)

amplitudes = np.abs(fourier)
phases = np.angle(fourier)

k = 5 # number of harmonic to use 
fourier[k:-k] = 0
fourier_inv = np.fft.ifft(fourier)

X_hat = np.rint(fourier_inv.real).astype(int)
Y_hat = np.rint(fourier_inv.imag).astype(int)
im_contour_hat = imPro.get_contour_image([X_hat, Y_hat])

fig, axs = plt.subplots(3,1)
axs[0].imshow(im_contour)
axs[1].imshow(im_contour_hat)
axs[2].plot(amplitudes,'x')

    
#%% Classification using FD
amplitudes_zeros = []
amplitudes_ones = []

for img in zeros:
    contour = imPro.get_outmost_contour(img)
    amplitudes = imPro.get_amplitude_first_descriptors(contour, n_descriptor = 4)
    amplitudes_zeros.append(amplitudes)
    
for img in ones:
    contour = imPro.get_outmost_contour(img)
    amplitudes = imPro.get_amplitude_first_descriptors(contour, n_descriptor = 4)
    amplitudes_ones.append(amplitudes)

amplitudes_zeros = np.array(amplitudes_zeros)
amplitudes_ones = np.array(amplitudes_ones)

fig, axs = plt.subplots(2,1, figsize = (8,12))

axs[0].plot(amplitudes_zeros[:,0], amplitudes_zeros[:,1],'.b', label = 'zeros')
axs[0].plot(amplitudes_ones[:,0], amplitudes_ones[:,1],'.r', label = 'ones')
axs[0].set_xlabel('Highest amplitude')
axs[0].set_ylabel('Second highest amplitude')
axs[0].set_title('Amplitude of Fourier descriptors')
axs[0].legend()

axs[1].plot(amplitudes_zeros[:,0], amplitudes_zeros[:,2],'.b', label = 'zeros')
axs[1].plot(amplitudes_ones[:,0], amplitudes_ones[:,2],'.r', label = 'ones')
axs[1].set_xlabel('Second highest amplitude')
axs[1].set_ylabel('Third highest amplitude')
axs[1].set_title('Amplitude of Fourier descriptors')
axs[1].legend()

plt.show()
