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
#!!!
#it makes no sens to define two functions wich do exactly the same...
#define load_img_seq(file_path) or something like that:

def load_img_seq(file_path):
    img_names = [nm for nm in os.listdir(file_path) if '.png' in nm]  # make sure to only load .png
    img_names.sort()  # sort file names
    ic = skimage.io.imread_collection([os.path.join(file_path, nm) for nm in img_names])
    img_seq = skimage.io.concatenate_images(ic)
    return img_seq
#!!!

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


def plot_fourier_descriptors(fourier):
    fix, axs = plt.subplots(2, 1, figsize=(5, 5))
    axs[0].plot(np.abs(fourier))
    axs[1].plot(np.angle(fourier))


#%% Edge extraction methods

def get_outmost_contour(img):
    """
    Image analysis lab 2 \n
    Returns the ordered contour (outside contour) of the number (in grayscale) given as parameter

    :param img:  grayscale image of a number

    :return: ordered [X,Y] np.array of all the pixels in the contour

    """
    tmp = skimage.morphology.area_closing(img,area_threshold=250)
    edgeMap = skimage.feature.canny(tmp)
    [X,Y] = np.where(edgeMap)
    return get_ordered_contour([X,Y], 0)


def get_ordered_contour(contour, starting_index):
    """
    Image analysis lab 2 - private method \n
    This function orders a given contour so that it is sorted in a way that every point is touching in the actual picture his neighboors.


    :param contour: [X,Y] np.array of all the pixels in the contour
    :param starting_index: index to get the first point of the array

    :return: sorted [X,Y] np.array of all the pixels in the contour

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

def get_amplitude_first_descriptors(contour, n_descriptor):
    """
    

    Parameters
    ----------
    contour :   [[ïnt]]
        ordered [X,Y] arrays containing all the contour pixels.
    n_descriptor : int
        number of the n first descriptors (or harmonics) to obtain the amplitudes of.

    Returns
    -------
    toReturn : [float]
        n * 1 array with the amplitudes of the n first descriptors, also in order.

    """
    
    [X,Y] = contour
    signal = X + 1j * Y
    fourier = np.fft.fft(signal)
    # fourier = normalize_fourier(fourier)

    amplitudes = np.abs(fourier)
    sorted_amplitudes = np.sort(amplitudes)
    toReturn = sorted_amplitudes[-n_descriptor:]
    toReturn = toReturn[np.arange(toReturn.size-1,-1,-1)]
    return toReturn

def normalize_fourier(fourier):
    """
    This function will perform a normalization of the Fourier Series.

    Parameters
    ----------
    fourier :  ndarray
        the fourier Series to be normalized

    Returns
    -------
    out: ndarray
        The normalized fourier Series
    """
    fourier[0] = 0
    fourier = fourier / np.abs(fourier[1])
    phi = np.angle(fourier[1])
    fourier = [f * np.exp(-1j * phi * k) for k, f in enumerate(fourier)]
    # fourier = np.fft.fftshift(fourier)
    return fourier

def rotation_invariance(fourier):
    indexOfMax = np.argmax(np.abs(fourier))
    phase = np.angle(fourier)[indexOfMax]
    fourier_normalised = fourier * np.exp(-1j*phase)
    return fourier_normalised

# %% Jonas' functions
"""
Created on Sa Apr 17 2020

@author: jtu
"""
# %% general purpose functions
def mirrored_periodisation(idx,N):
    ''' mirrored periodisation of signal of length N evaluated at index current_idx
        For a signal s={...0,0,|0|,1,2,3,0,0,...} the result is
        s_p={...2,1,|0|,1,2,3,2,1,0,1,...}
        \param N        support of signal, in example above N=4
        \param idx      index where to evaluate periodic version s_p ot s
    '''
    if idx<0:
        idx=-idx
    elif idx>=N:
        #try and write it out if you want to understand...
        #...it is just a mathematical expression for a mirrored periodic sequence of indices
        idx=(int(idx/(N-1))%2)*(N-1-idx%(N-1))+(1-int(idx/(N-1))%2)*(idx%(N-1))

    #recursive correction
    if idx<0 or idx>=N:
        idx=mirrored_periodisation(idx,N);

    #retunr equivalent index in bounds [0;N-1]
    return idx

def get_nbh(img,x,y,w=3,h=3):
    ''' get neighbourhood around x,y pixel with mirrored periodisation, works
        only with odd w,h! no test if odd or even, so be careful!
        \param w,h      width of nbh, height of nbh. Must be odd
        \param x,y      position where nbh is extracted
    '''
    nx,ny=img.shape
    delta_w=int(w/2)
    delta_h=int(h/2)

    nbh=np.zeros((w,h))
    for xx in range(x-delta_w,x+delta_w+1):
        for yy in range(y-delta_h,y+delta_h+1):
            #get equivalent idx in case of values out of bounds
            x_idx=mirrored_periodisation(xx,nx)
            y_idx=mirrored_periodisation(yy,ny)
            #put values in nbh, origin (x,y) in the middle of nbh
            nbh[xx-x+delta_w][yy-y+delta_h]=img[x_idx][y_idx]

    return nbh

def load_img_seq(file_path, format='.png'):
    img_names = [nm for nm in os.listdir(file_path) if format in nm]  # make sure to only load .png
    img_names.sort()  # sort file names
    ic = skimage.io.imread_collection([os.path.join(file_path, nm) for nm in img_names])
    img_seq = skimage.io.concatenate_images(ic)
    return img_seq

# %% object description functions
def com(img):
    ''' computes center of mass of image, taking pixel values as weights and x,y
        pixel coordinates as the weights' position
    '''
    nx,ny=img.shape
    mid_x=int(nx/2)
    mid_y=int(ny/2)
    M=0
    x_M=0
    y_M=0
    m_i=0

    for x in range(nx):
        for y in range(ny):
            m_i=img[x,y];
            M+=m_i;
            x_M+=(x-mid_x)*m_i;
            y_M+=(y-mid_y)*m_i;

    return (x_M/M,y_M/M)

def object_covar_mat(img):
    ''' computes covariance matrix for a given image
    '''
    x_bar,y_bar=com(img)
    nx,ny=img.shape
    #generate x and y grids
    x = np.linspace(-nx/2, nx/2, nx)
    y = np.linspace(-ny/2, ny/2, ny)
    xv, yv = np.meshgrid(x, y)
    #substract center of mass
    xv-=x_bar
    yv-=y_bar
    #set to zero non object coordinates
    xv[img==0]=0
    yv[img==0]=0
    #compute covariance matrix
    sigma_xx=np.sum(xv**2)
    sigma_yy=np.sum(yv**2)
    sigma_xy=np.sum(xv*yv)

    return np.array([[sigma_xx, sigma_xy],[sigma_xy, sigma_yy]])

def compute_principal_angle(covar_mat):
    ''' computes the angle from the reference frame to the principal axis of object
    '''
    return np.arctan2(2*covar_mat[0,1],covar_mat[0,0]-covar_mat[1,1])/2

# %% morphological operations on stacks and img
def skel_and_thld(img):
    #thld img
    img[img>0]=1
    #skeletonize
    return morph.skeletonize(img)

def skel_img_stack(img_stack):
    skel_stack=[]
    for img in img_stack:
        #thld img
        img[img>0]=1
        #skeletonize
        skel_stack.append(morph.skeletonize(img))

    return skel_stack

# %% distance map creation
def dist_map(img,direct_dist=3,diag_dist=4):
    ''' computes the distance map for an object, settin the objects pixels at 0
        and the others to the equivalent of the smallest distance from that position
        to the object using a 8-connect method for the distance measurement
        distance convention: | diag_dist   direct_dist  diag_dist   |   |4 3 4|
                             | direct_dist      0       direct_dist | = |3 0 3|
                             | diag_dist   direct_dist  diag_dist   |   |4 3 4|
        by default
    '''
    nx,ny=img.shape
    #init output
    out=img.copy()
    out[img!=0]=0
    out[img==0]=img.size*diag_dist
    #define distance map
    dist_m=np.array([[diag_dist,direct_dist,diag_dist],[direct_dist,0,direct_dist],[diag_dist,direct_dist,diag_dist]])
    #8-connect mask
    eval_forward=np.array([[True,True,False],[True,True,False],[True,False,False]])
    eval_backward=np.array([[False,False,True],[False,True,True],[False,True,True]])

    #scan image forwards:
    for x in range(nx):
        for y in range(ny):
            nbh=get_nbh(out,x,y)
            nbh+=dist_m
            out[x,y]=np.amin(nbh[eval_forward])
    #scan image backwards
    for x in range(nx-1,-1,-1):
        for y in range(ny-1,-1,-1):
            nbh=get_nbh(out,x,y)
            nbh+=dist_m
            out[x,y]=np.amin(nbh[eval_backward])

    return out
