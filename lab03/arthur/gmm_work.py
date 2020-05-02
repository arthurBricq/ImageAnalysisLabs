#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:19:31 2020

@author: arthur
"""

import sys
sys.path.append("..")
import lab03_functions as imPro 
import matplotlib.pyplot as plt 
import importlib
importlib.reload(imPro)

import numpy as np
import itertools
from sklearn import mixture


# %% Plot the training data sets (there is a function for this too)

a,b,c, = imPro.get_matlab_data()
datasets = [a,b,c]
fig, ax = plt.subplots() 
for data in datasets:
    ax.plot(data[:,0],data[:,1],'.')
fig.suptitle('Training datasets for lab03')
    

# %% Selection of the best fit --> 1 componen, diagonal.
X = a
bic = []
lowest_bic = np.infty
n_components_range = range(1, 7)
cv_types = ['spherical',  'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue'])
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):(i + 1) * len(n_components_range)], width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +.2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)

# %% mahalanobis distance 

def mahalanobis(gmm, points):
    """
    Compute the mahakanobis distance of a dataset of points from a GMM

    Parameters
    ----------
    gmm : GMM 
        GMM of the model.
    points : np.array of size (N,2) where N is the number of point 
        All the data points to be used.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    mean = gmm.means_[0] 
    cov = np.diag(gmm.covariances_[0])
    # cov = np.eye(2) 
    v = (points - mean).T
    y = v.T.dot(np.linalg.inv(cov))    
    scalar = (y * v.T).sum(axis = 1)
    return np.sqrt(scalar)


# %% Final code

datasets = [a,b]

# 1. Split the data 
training_sets, testing_sets = [], []
for dataset in datasets: 
    training, testing = imPro.split_dataset(dataset = dataset, training_ratio = 0.7)
    training_sets.append(training)
    testing_sets.append(testing)
    
# 2. Fit the GMM on the training datasets
gmms = []
for dataset in training_sets:
    gmm = mixture.GaussianMixture(n_components=1, covariance_type='diag')
    gmm.fit(dataset)
    gmms.append(gmm)

# 3. Test: evaluate Mahalanobis distance to perform classification of testing points
classification_results = []
for i, dataset in enumerate(testing_sets):
    distances = np.array([mahalanobis(gmm, dataset) for gmm in gmms])
    predicts = np.argmin(distances, axis = 0)
    corrects = predicts == i 
    classification_results.append(corrects)
    
classification_results = np.array(classification_results)
non_zero = np.count_nonzero(classification_results)
elements = classification_results.size
p1 = np.count_nonzero(classification_results[0,:])/classification_results[0,:].size
p2 = np.count_nonzero(classification_results[1,:])/classification_results[1,:].size
print("The overall accuracy of the model is {} successes over {} testing samples".format(non_zero, elements))

# 4. Plotting the results 

fig, ax = plt.subplots()
colors = ['turquoise', 'darkorange']
imPro.make_ellipses(gmms[0], ax, colors[0])
imPro.make_ellipses(gmms[1], ax, colors[1])
colors = ['tab:blue', 'darkorange']
for i, dataset in enumerate(testing_sets):
    for j, data in enumerate(dataset):
        f = '.' if classification_results[i,j] else 'x'
        ax.plot(data[0],data[1],f,color = colors[i])
ax.set_title('Classification results on the testing set. \n \'x\' stands for missclassification')
x, y = ax.get_xlim()[0], ax.get_ylim()[1]+1
ax.text(x + 1, y, "Class 1: {} success rate".format(p1),{'color':'gray'})
ax.text(x + 1, y - 1, "Class 2: {} success rate".format(p2),{'color':'gray'})

