# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 18:06:39 2020

@author: abric
"""


import skimage.io
import matplotlib.pyplot as plt
import skimage.morphology as morph
import skimage.filters as flt
import numpy as np


### PLOTTING FUNCTIONS

def plotImage(image):
    """
    Plot the image
    """
    plt.figure()
    plt.imshow(image)
    
def plotHistogram(image):
    """
    Plot the histogram of the image
    """
    plt.figure()
    plt.hist(image.ravel(), bins=256, histtype='step', color='black')
    
def plotEdgeMap(edgeMap):
    nx,ny=edgeMap.shape
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(edgeMap, cmap='gray')
    ax.set_title('Contours ({} px, {} px)'.format(nx, ny))

    
    
### LABELIZATION ALGORITHM
# This is the algorithm that groups pixels within parts of the picture where the boundary is closed.
# It attributes labels to each pixel, standing for each group.

def labelizePixels(edgeMap,minValue=0):
    """
    ------ Image analysis lab 1 --------
    Takes an edge map image as input and returns the map of labels such that every pixel 
    belongs to a cluster defined by the edge detection.
    This is therefore a labelization algorithm.
    INPUT:
        - image: image with edged pixels (with -1 or less for each pixel)
        - minValue: int under which the pixels need to be detected as edges in the edge map
    REQUIREMENT:
        "all edges marked with negative values"
        "all the rest marked with 0"
    OUTPUT:
        - labels: label of each pixel
    """
    image = edgeMap
    im_h,im_w = image.shape
    labels = np.zeros(shape=(im_h,im_w))
    currentLabel = 0
    buffer = [(0,0)] 
    while True:
        currentLabel += 1
        while len(buffer) > 0:
            x,y = buffer[0]
            buffer = buffer[1:]
            if image[x, y] < minValue: continue
            labels[x, y] = currentLabel
            if (x-1 >= 0) and (labels[x-1, y] == 0) and (x-1,y) not in buffer:
                    buffer.append((x-1,y))
            if (x+1 < im_h) and (labels[x+1, y] == 0) and (x+1,y) not in buffer:
                    buffer.append((x+1,y))
            if (y-1 >= 0) and (labels[x, y-1] == 0) and (x,y-1) not in buffer:
                    buffer.append((x,y-1))
            if (y+1 < im_w) and (labels[x, y+1] == 0) and (x,y+1) not in buffer: 
                    buffer.append((x,y+1))
            
        # If this point is reached, it means the last buffer is completed
        # and now we need to start a new one ! 
        # To find it: the first point 
        cond = np.logical_and(image >= minValue, labels == 0)
        indexes = np.argwhere(cond)
        if indexes.shape[0] == 0:
            break
        buffer = [indexes[0,:]]
    
    return labels

### CLASSIFICATION ALGORITHM
# to find the number of shapes present in the labelled pixel map.    

def labels2Shapes(labels):
    """
    Image Analysis, lab 1 part II 
    This function is used to count the number of shapes present in a picture.
    It takes as input 'labels', which is a Label map generated by the previous function 'labelizePixels'
    It returns as output an array groups of np.array, each one of them containing the different shapes found.
    """
    uniques, occurences = np.unique(labels,return_counts = True)
    # Remove the background, the borders, and the values that are too small to be relevant
    sortedIndexes = np.argsort(occurences)
    occurences = occurences[sortedIndexes]
    occurences = occurences[:-2]
    minimumAmountOfPixel = 20 
    occurences = occurences[occurences>minimumAmountOfPixel]
    
    # While loop to cluster the occurences by groups
    # The groups are the shapes 
    # The number of element per loops is the number of elements of each shape on the picture
    groups = []
    allowedRange = 175
    # While occurence is not empty
    while len(occurences):
        # Get the first value of occurences
        value = occurences[0]
        # Get the neighboors and add them as a group
        neighboorsIndex = np.abs(occurences - value)<allowedRange
        groups.append(occurences[neighboorsIndex])
        # And remove them from occurences
        occurences = occurences[np.logical_not(neighboorsIndex)] 
        
    return groups


### FUNCTIONS FOR COUNTING NUMBER OF SHAPES

def countNumberOfShapes(im_bi):
    """
    Given a binary edge map, this function will count the number of shapes present in this edge map.
    It uses the number of pixel in each closed region to differentiate the shapes between them. 
    See the 2 functions used:
        - labelizePixel, to form the closed regions with labels
        - labels2shapes, that does the counting job of the shapes
    """
    
    # Apply gaussian fitler to finish the end 
    im_bi = skimage.filters.gaussian(im_bi,sigma=1) * 255
        
    # Apply any border detection 
    edgeMap = skimage.filters.laplace(im_bi)
    edgeMap[edgeMap>0]=-edgeMap[edgeMap>0]
    
    # Clustering
    labels = labelizePixels(edgeMap, minValue = -4)
    
    # Analysis of the labels      
    return labels2Shapes(labels)

    
def labels2Centers(labels):
    """
    Image Analysis, lab 1 part II 
    This function is used to count the number of shapes present in a picture.
    It takes as input 'labels', which is a Label map generated by the previous function 'labelizePixels'
    It returns as output an array groups of np.array, each one of them containing the different shapes found.
    """
    uniques, occurences = np.unique(labels,return_counts = True)
    
    # Remove the background, the borders, and the values that are too small to be relevant
    sortedIndexes = np.argsort(occurences)
    uniques, occurences = uniques[sortedIndexes], occurences[sortedIndexes]
    uniques, occurences = uniques[:-2], occurences[:-2]
    minimumAmountOfPixel = 20 
    indexes = occurences>minimumAmountOfPixel
    uniques, occurences = uniques[indexes], occurences[indexes]
    
    centers = []
    for label in uniques:
        positions=np.argwhere(labels==label)
        center = positions.mean(axis=0)
        centers.append(center)
        
    return np.array(centers)    
    

def findCentersOfClusters(im_bi):
    # Apply gaussian fitler to finish the end 
    im_bi = skimage.filters.gaussian(im_bi,sigma=1) * 255
        
    # Apply any border detection 
    edgeMap = skimage.filters.laplace(im_bi)
    edgeMap[edgeMap>0]=-edgeMap[edgeMap>0]
    
    # Clustering
    labels = labelizePixels(edgeMap, minValue = -4)
    
    return labels2Centers(labels)

def blueThresholdForPixel(pixel, image):
    """
    Takes on pixel and on image, and make a threshold for the value in the blue channel around this given pixel.
    Returns the thresholded image 
    """
    x_interest, y_interest = pixel
    B = image[x_interest,y_interest]
    deltaB = 100
    testB = np.logical_and(image[:,:]<B+deltaB,image[:,:]>B-deltaB)
    return testB


### FUNCTIONS FOR EDGE DETECTION 2
    
def skeletonize_based_sobel(img,sigma,t1,t2,G_t1,G_t2,input_thld=True):
    ''' edge detection using morphological approach. Good for big features with strong edges
        \param img          input image to extract contours
        \param t1,t2        upper and lower threshold to choose region/pixels of interest
                            in input image, t1<pixels of intrest<t2
        \param G_t1,G_t2    upper and lower threshold to choose region/pixels of interest
                            in gradient image G_t1<pixels of intrest<G_t2
        \param input_thld   if "True" applies t1,t2, if "False" set t1=t2=0 or random value...
        \return (edges,thld_img,G,Phi,G_thld_img)
    '''
    nx,ny=img.shape
    
    #threshold image to highlight region of interest
    if input_thld:
        thld_img=255*np.ones((nx,ny))
        idx_m=img<t1
        idx_p=img>t2
        idx=idx_p+idx_m #logic and
        thld_img[idx]=0
        
        #refine desired area
        thld_img=morph.closing(thld_img,np.array([[0,0,0],[0,1,1],[0,1,0]]))
    else:
        thld_img=img
        
    #smoothing
    thld_img=flt.gaussian(thld_img,sigma)

    #compute magnitude and phase of gradient for thresholded image
    (G,Phi)=gradient_sobel(thld_img)#,sigma) #slow part of algorithm!!!

    #threshold gradient magnitude
    G_thld_img=np.ones((nx,ny))
    idx_m=G<G_t1
    idx_p=G>G_t2
    idx=idx_p+idx_m #logic and
    G_thld_img[idx]=0

    #extract edges
    edges=morph.skeletonize(G_thld_img)

    return (edges,thld_img,G,Phi,G_thld_img)

def gradient_sobel(img):
    ''' computation of gradien magnitude and phase using sobel filters
    '''
    gradX=flt.sobel(img) 
    gradY=flt.sobel_v(img) 
    
    nx,ny=img.shape
    G=np.zeros((nx,ny))
    Phi=np.zeros((nx,ny))
    
    for x in range(nx):
        for y in range(ny):
            G[x][y]=np.sqrt(gradX[x][y]**2+gradY[x][y]**2)
            Phi[x][y]=np.arctan2(gradY[x][y],gradX[x][y])
    
    return (G, Phi)

def minmax(a,scale=1):
    ''' minmax normalisation puts values [min,max]-->[0,1]
        \param a            input array to normalize
        \param scale        scales to [0,scale] the normalized intervall
    '''
    nx,ny=a.shape
    min_val=np.amin(a)
    max_val=np.amax(a)
    out=a.copy()
    
    for x in range(nx):
        for y in range(ny):
            out[x][y]=scale*(a[x][y]-min_val)/(max_val-min_val)
            
    return out

### FUNCTIONS FOR REGION GROWING

def homogeneityCond(pixel,neighbour,maxdiff):
    """
    Condition for homogeneity. used for regionGrowing.
    """
    return np.abs(int(pixel)-int(neighbour))<maxdiff

def regionGrowing(image, start_coord, maxdiff,dynamic_homogeneity=False):
    """
    Does the region growing from start_coord in image, with 4-connectivity.
    Homogeneity condition is either dynamic : comparison of pixel value to its neighbour
        or static : comparison of pixel value to value of pixel of start_coord
        maxdiff is the maximum difference between the two pixel values
    """
    im_w,im_h=image.shape
    visitedPoints = np.zeros(shape=(im_h,im_w))
    region_size = 1
    buffer = [start_coord]
    while len(buffer) > 0:
        
        x,y = buffer[0]
        if (x>=256 or x<0 or y>=256 or y<0): 
            continue
        visitedPoints[x,y] = 1
        #print ("point x, y :", x, "  ",y)
        #print (buffer)
        buffer = buffer[1:]
        if dynamic_homogeneity :
            if  homogeneityCond(image[x, y],image[x-1,y],maxdiff):
                if visitedPoints[x-1,y]==0 and (x-1,y) not in buffer :
                    buffer.append((x-1,y))
                    region_size+=1
            if homogeneityCond(image[x, y],image[x+1,y],maxdiff) :
                if visitedPoints[x+1,y]==0 and (x+1,y) not in buffer :
                    buffer.append((x+1,y))
                    region_size+=1
            if homogeneityCond(image[x, y],image[x,y-1],maxdiff) :
                if visitedPoints[x,y-1]==0 and (x,y-1) not in buffer :
                    buffer.append((x,y-1))
                    region_size+=1
            if homogeneityCond(image[x, y],image[x,y+1],maxdiff) :
                if visitedPoints[x,y+1]==0 and (x,y+1) not in buffer :
                    buffer.append((x,y+1))
                    region_size+=1
        else :
            if  homogeneityCond(image[start_coord],image[x-1,y],maxdiff):
                if visitedPoints[x-1,y]==0 and (x-1,y) not in buffer :
                    buffer.append((x-1,y))
                    region_size+=1
            if homogeneityCond(image[start_coord],image[x+1,y],maxdiff) :
                if visitedPoints[x+1,y]==0 and (x+1,y) not in buffer :
                    buffer.append((x+1,y))
                    region_size+=1
            if homogeneityCond(image[start_coord],image[x,y-1],maxdiff) :
                if visitedPoints[x,y-1]==0 and (x,y-1) not in buffer :
                    buffer.append((x,y-1))
                    region_size+=1
            if homogeneityCond(image[start_coord],image[x,y+1],maxdiff) :
                if visitedPoints[x,y+1]==0 and (x,y+1) not in buffer :
                    buffer.append((x,y+1))
                    region_size+=1       
    plt.figure()
    plt.imshow(image)
    plt.imshow(visitedPoints, alpha=0.5)
    print ("Region size : ",region_size)
    print ("Ratio : ",region_size/im_w/im_h)
    return region_size



