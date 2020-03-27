# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:59:45 2020

@author: abric
"""

import skimage.io
import matplotlib.pyplot as plt
import os
import numpy as np 

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