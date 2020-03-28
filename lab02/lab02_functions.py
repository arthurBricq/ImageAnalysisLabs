# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:59:45 2020

@author: abric
"""

import skimage.io
import matplotlib.pyplot as plt
import os
import numpy as np 
import skimage, skimage.feature, skimage.morphology


#%% Functions to get the data

def get_zeros():
    #  Load zeros
    zeros_path = 'lab-02-data/part1/0'
    zeros_names = [nm for nm in os.listdir(zeros_path) if '.png' in nm]  # make sure to only load .png
    zeros_names.sort()  # sort file names
    ic = skimage.io.imread_collection([os.path.join(zeros_path, nm) for nm in zeros_names])
    zeros_im = skimage.io.concatenate_images(ic)
    return zeros_im

def get_ones():
    #  Load ones
    ones_path = 'lab-02-data/part1/1'
    ones_names = [nm for nm in os.listdir(ones_path) if '.png' in nm]  # make sure to only load .png
    ones_names.sort()  # sort file names
    ic = skimage.io.imread_collection(([os.path.join(ones_path, nm) for nm in ones_names]))
    ones_im = skimage.io.concatenate_images(ic)
    
    return ones_im

#%% Plotting function
    
def get_contour_image(contour):
    [X,Y] = contour
    im = np.zeros((28, 28))     
    im[X,Y] = 1 
    return im
    
    

#%% Edge extraction methods

def get_outmost_contour(img, starting_index = 0):
    """
    Image analysis lab 2
    Returns the ordered contour (outside contour) of the number (in grayscale) given as parameter
    
    Parameters
    -----------
    image : grayscale image of a number
    
    Returns 
    -------
    contour : ordered [X,Y] np.array of all the pixels in the contour
    
    """
    tmp = skimage.morphology.area_closing(img,area_threshold=250)
    edgeMap = skimage.feature.canny(tmp)
    [X,Y] = np.where(edgeMap)
    return get_ordered_contour([X,Y], starting_index)


def get_ordered_contour(contour, starting_index):
    """
    Image analysis lab 2 - private method
    This function orders a given contour so that it is sorted in a way that every point is touching in the actual picture his neighboors.
    
    Parameters
    ------------
    contour : [X,Y] np.array of all the pixels in the contour
    starting_index : index to get the first point of the array
    
    Returns
    --------
    contour : sorted [X,Y] np.array of all the pixels in the contour
    
    """
    [X_old,Y_old] = contour
    X,Y = [],[]
    
    # Get the starting x,y
    x, y = X_old[starting_index], Y_old[starting_index]
    i = starting_index
    
    while len(X_old)-1:
        # Update the arrays
        X.append(x)
        Y.append(y)
        X_old, Y_old = np.delete(X_old, i), np.delete(Y_old, i)
        
        # Find the next x,y
        distance = np.sqrt( (X_old-x) ** 2 + (Y_old-y) ** 2 )
        i = np.argmin(distance)
        x, y = X_old[i], Y_old[i]
        
    X.append(x) ; Y.append(y) 
    X, Y = np.array(X), np.array(Y)
    return [X,Y]

    


#%% Fourier Descriptors 
    
def get_amplitude_first_descriptors(contour,n_descriptor):
    """
    Return the amplitude of the N first fourier descriptor for the given contour
    
    Parameters
    ----------
    contour : **ordered** [X,Y] arrays containing all the contour pixels
    n_descriptors : number of the n first descriptors (or harmonics) to obtain the amplitudes of.
    
    RETURNS
    --------
    amplitudes : n * 1 array with the amplitudes of the n first descriptors, also in order
    """
    [X,Y] = contour
    signal = X + 1j * Y
    fourier = np.fft.fft(signal)
    amplitudes = np.abs(fourier)
    sorted_amplitudes = np.sort(amplitudes)
    toReturn = sorted_amplitudes[-n_descriptor:]
    toReturn = toReturn[np.arange(toReturn.size-1,-1,-1)]
    return toReturn
    