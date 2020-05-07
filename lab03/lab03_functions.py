
import os
import scipy.io
import matplotlib.pyplot as plt 


import sys
sys.path.append("..")
import lab03_functions as imPro 
import matplotlib as mpl
import importlib
importlib.reload(imPro)

import numpy as np
import itertools
from sklearn import mixture


#%% Data functions

def get_matlab_data():
    """
    Returns the matlab data as numpy array for the lab 3

    Returns
    -------
    a : 200x2 array
        Data array for the lab.
    b : 200x2 array
        Data array for the lab.
    c : 200x2 array
        Data array for the lab.
        
    """
    data_folder = 'lab-03-data'
    data_part1_path = os.path.join(data_folder, 'part1', 'classification.mat')
    matfile = scipy.io.loadmat(data_part1_path)
    a = matfile['a']
    b = matfile['b']
    c = matfile['c']
    return a,b,c


def plot_training_datasets():
    a,b,c, = get_matlab_data()
    datasets = [a,b,c]
    fig, ax = plt.subplots() 
    for data in datasets:
        ax.plot(data[:,0],data[:,1],'.')
    fig.suptitle('Training datasets for lab03')
        
    
# %% GMM functions 


def split_dataset(dataset, training_ratio):
    """
    Returns the splitted dataset, into a training dataset and a testing one

    Parameters
    ----------
    dataset : np.array of size N x M 
        Initial data set.
    training_ratio : float
        Percentage of data to be used in the training set.

    Returns
    -------
    [training_set, testing_set]
        The splitted arrays.

    """
    np.random.shuffle(dataset)
    return np.split(dataset, [int(dataset.shape[0]*training_ratio)], axis = 0)


def make_ellipses(gmm, ax, color):
    covariances = np.diag(gmm.covariances_[0])
    v, w = np.linalg.eigh(covariances)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse(gmm.means_[0, :2], v[0], v[1],180 + angle, color=color)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.5)
    ax.add_artist(ell)
    ax.set_aspect('equal', 'datalim')
    
    
def arr2vec(arr):
    ''' converts array to vector
    '''
    vec=np.reshape(arr, arr.size)
    return vec

def rearrange_img_stack(img_stack,z_stack=True):
    ''' arranges a image stack in a 2d array where each row is an image
    '''
    if z_stack:
        nx,ny,nz=img_stack.shape
    else:
        nz,nx,ny=img_stack.shape

    img_arr=np.zeros((nz,nx*ny))

    for z in range(nz):
        if z_stack:
            img_arr[z,:]=arr2vec(img_stack[:,:,z])
        else:
            img_arr[z,:]=arr2vec(img_stack[z])
    return img_arr
