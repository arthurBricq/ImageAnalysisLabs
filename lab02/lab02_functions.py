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
import skimage.morphology as morph


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


def plot_fourier_descriptors(fourier):
    fix, axs = plt.subplots(2, 1, figsize=(5, 5))
    axs[0].plot(np.abs(fourier))
    axs[1].plot(np.angle(fourier))


#%% Contour extraction methods

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

def get_amplitude_first_descriptors(fourier, n_descriptor = 4):
    """
    Returns the N-higesth amplitudes of the fourier descriptors

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

    amplitudes = np.abs(fourier)
    sorted_amplitudes = np.sort(amplitudes)
    toReturn = sorted_amplitudes[-n_descriptor:]
    toReturn = toReturn[np.arange(toReturn.size-1,-1,-1)]
    return toReturn

def rotation_invariance(fourier):
    """
    Take the phase of the descriptor with higest amplitude and substract it to all other phases.

    Parameters
    ----------
    fourier : TYPE
        DESCRIPTION.

    Returns
    -------
    fourier_normalised : TYPE
        DESCRIPTION.

    """
    indexOfMax = np.argmax(np.abs(fourier))
    phase = np.angle(fourier)[indexOfMax]
    fourier_normalised = fourier * np.exp(-1j*phase)
    return fourier_normalised

def scaling_invariance(fourier):
    r1 = np.abs(fourier[1])
    fourier = fourier / r1
    return fourier

def starting_point_invariance(fourier):
    phase1 = np.angle(fourier[1])
    fourier = [f * np.exp(-1j * phase1 * k) for k, f in enumerate(fourier)]
    return fourier

def translation_invariance(fourier):
    fourier[0] = 0
    return fourier


def get_feature_vector(img, invariances):
    """
    Returns the feature vector for the given image with the given invariances applied to the Fourier descriptors.


    Parameters
    ----------
    img :
        Image to be analysed (from this image, the contour is extracted and then Fourier Descriptors are created)
    invariances :
        Array of four boolean elements, as follow [mathrm{Re}]

    Returns
    -------
    x :
        The feature vector of our Fourier Descriptors

    """
    rot, scal, trans, SP = invariances
    [X,Y] = get_outmost_contour(img)
    signal = X + 1j * Y
    fourier = np.fft.fft(signal)

    if rot: fourier = rotation_invariance(fourier)
    if scal: fourier = scaling_invariance(fourier)
    if trans: fourier = translation_invariance(fourier)
    if SP: fourier = starting_point_invariance(fourier)

    f0 = fourier[0]
    f1 = fourier[1]
    f2 = fourier[2]
    f1n= fourier[-1]

    x = [f1.real, np.abs(f1), np.abs(f1n),np.abs(f1-f1n),f2.real,f0.real, f0.imag]
    return x



#%% Plotting functions for Illustrating Fourier Descriptors

def plot_FD_rotation_invariance(img, theta, ax):
    """
    Make a plot of the Fourier Descriptors' phases of the contour and of the rotated contour

    Parameters
    ----------
    img :
        Image.
    theta :
        Angle of the rotation.
    ax :
        Where to plot the data.

    Returns
    -------
    None.

    """
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    X1, Y1 = get_outmost_contour(img)
    contour1 = np.column_stack([X1, Y1])
    contour2 = contour1 @ R
    X2, Y2 = contour2[:, 0], contour2[:, 1]
    signal1 = X1 + 1j * Y1
    fourier1 = np.fft.fft(signal1)
    signal2 = X2 + 1j * Y2
    fourier2 = np.fft.fft(signal2)

    ax.plot(np.angle(fourier1), '-.r', label = 'contour')
    ax.plot(np.angle(fourier2),'-.b', label = 'rotated contour')
    ax.set_title('$\\theta$  =  {:.3} rad'.format(theta))
    ax.legend()

def plot_FD_scaling_invariance(img, alpha, ax):
    """
    Make a plot of the Fourier Descriptors' amplitude of the contour and of the scalled contour


    Parameters
    ----------
    img :
        Image
    alpha :
        Scalling factor
    ax :
        Where to plot

    Returns
    -------
    None.

    """
    R = np.array([[alpha, 0], [0, alpha]])
    X1, Y1 = get_outmost_contour(img)
    contour1 = np.column_stack([X1, Y1])
    contour2 = contour1 @ R
    X2, Y2 = contour2[:, 0], contour2[:, 1]
    signal1 = X1 + 1j * Y1
    fourier1 = np.fft.fft(signal1)
    signal2 = X2 + 1j * Y2
    fourier2 = np.fft.fft(signal2)

    ax.plot(np.abs(fourier1)[1:5], '-.r', label = 'Contour')
    ax.plot(np.abs(fourier2)[1:5],'-.b', label = 'Scaled contour')
    ax.set_title('$\\alpha$ = {}'.format(alpha))
    ax.set_xlabel('First 5 harmonics')
    ax.legend()


#%% Class used as Plotting data model


class PlotData:
    """
    This class represents generic data to make 1 plot of 2 features of all images.
    I have used this class to make a nice plot which really illustrates easily different cases.
    The array 'invariances' is used within the function 'get_feature_vector'
    """
    def __init__(self, name, invariances, features):
        self.name = name
        self.invariances = invariances
        self.features = features

    def get_features_label(self):
        l1 = self._get_feature_label(self.features[0])
        l2 = self._get_feature_label(self.features[1])
        return [l1,l2]


    def _get_feature_label(self,feature):
        if feature == 0: return "$\\mathrm{Re}f_1$"
        if feature == 1: return "$|f_1|$"
        if feature == 2: return "$|f_{-1}|$"
        if feature == 3: return "$|f_1-f_{-1}|$"
        if feature == 4: return "$\\mathrm{Im}f_2$"
        if feature == 5: return "$\\mathrm{Re}f_0$"
        if feature == 6: return "$\\mathrm{Im}f_0$"

# %% image stack plotting functions
def plot_img_stacks(stack_list, nz=1, fig_size=None):
    ''' plots multiples image stacks with nz frames stored in list
    '''
    n_stacks=len(stack_list)

    if fig_size is not None:
        fig, axes = plt.subplots(n_stacks, nz, figsize=fig_size)
    else:
        fig, axes = plt.subplots(n_stacks, nz)

    for i in range(n_stacks):
        for j in range(nz):
            ax=axes[i][j];
            im=stack_list[i][j]
            ax.imshow(im, cmap='gray')
            ax.axis('off')

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


def binarize(image):
    t =skimage.filters.threshold_mean(image)
    return image>t

#%% Region based Descriptors

def area(image):
    """
    Returns the area of the number
    :param image:  grayscale image of a number (white number on black background)
    """
    area=np.count_nonzero(image)
    return area


def perimeter(image):
    """
    Returns the perimeter of the number
    :param image:  grayscale image of a number (white number on black background)
    """
    perimeter=np.array(get_outmost_contour(image)).shape[1]
    return perimeter

def area_polygon(image):
    """
    Returns the area in the polygon connecting the contour points of the number
    :param image:  grayscale image of a number (white number on black background)
    """
    nb_inside= area(image)-perimeter(image)
    areap= perimeter(image)/2 +nb_inside -1
    return areap

def compacity(image):
    """
    Returns the compacity of the number
    :param image:  grayscale image of a number (white number on black background)
    """
    return perimeter(image)**2/area(image)

def projection(image):
    proj_x=[]
    proj_y=[]
    for i in range(image[0].shape[0]) :
        proj_y.append(np.count_nonzero(image[i]))
    for j in range (image[1].shape[0]):
        proj_x.append(np.count_nonzero(image[j]))
    return proj_x,proj_y

def moment (image,i,j) :
    m= 0
    for k in range(image[0].shape[0]):
        for l in range(image[0].shape[0]):
            m=m+pow(k,i)*pow(l,j)*image[k,l]/256
    return m

def centers_gravity (image):
    kc=moment(image,1,0)/moment(image,0,0)
    lc=moment(image,0,1)/moment(image,0,0)
    return [kc,lc]

def centered_moments(image,i,j,scaling_invariant=False) : #invariant to translation : center of gravity as origin
    mc= 0
    for k in range(image[0].shape[0]):
        for l in range(image[0].shape[0]):
            mc=mc+pow(k-centers_gravity(image)[0],i)*pow(l-centers_gravity(image)[1],j)*image[k,l]/256
    if scaling_invariant :
        gamma=int((i+j)/2)+1
        mc=centered_moments(image,i,j)/centered_moments(image,0,0)**gamma
    return mc

def standard_centered_moments(image,order=1,scaling_invariant=False):
    if (order==1):
        return centered_moments(image,2,0,scaling_invariant)+centered_moments(image,0,2,scaling_invariant)
    if (order==2):
        return (centered_moments(image,2,0,scaling_invariant)-centered_moments(image,0,2,scaling_invariant))**2 +4*centered_moments(image,1,1,scaling_invariant)
    if (order==3):
        return (centered_moments(image,3,0,scaling_invariant)-3*centered_moments(image,1,2,scaling_invariant))**2 + (3*centered_moments(image,2,1,scaling_invariant)-centered_moments(image,0,3,scaling_invariant))**2
    if (order==4) :
        return (centered_moments(image,3,0,scaling_invariant)+centered_moments(image,1,2,scaling_invariant))**2 + (centered_moments(image,2,1,scaling_invariant)+centered_moments(image,0,3,scaling_invariant))**2
    return



def region_based_features(images):
    """
    Returns an array of region based features for all the images in the input

    :param images:  group of grayscale images of numbers
    :return: param_images np.array of all the region based features of the input images
    """
    param_images = []
    for img in images:
        binary=binarize(img)
        param=[]
        param.append(area(binary))
        param.append(perimeter(img))
        param.append(compacity(img))
        #param.append(standard_centered_moments(img,1,False))
        #param.append(standard_centered_moments(img,1,True))
        param_images.append(param)
    param_images = np.array(param_images)
    return param_images
