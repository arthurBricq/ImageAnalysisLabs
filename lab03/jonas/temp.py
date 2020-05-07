# %% markdown
# # [IAPR 2020:][iapr2020] Lab 3 ‒  Classification
#
# **Author:** first_name_1 last_name_1, first_name_2 last_name_2, first_name_3 last_name_3
# **Due date:** 08.05.2020
#
# [iapr2018]: https://github.com/LTS5/iapr-2018
# %% markdown
# ## Extract relevant data
# We first need to extract the `lab-03-data.tar.gz` archive.
# To this end, we use the [tarfile] module from the Python standard library.
#
# [tarfile]: https://docs.python.org/3.6/library/tarfile.html
# %% codecell
import tarfile
import os

data_base_path = os.path.join(os.pardir, 'data')
data_folder = 'lab-03-data'
tar_path = os.path.join(data_base_path, data_folder + '.tar.gz')
with tarfile.open(tar_path, mode='r:gz') as tar:
    tar.extractall(path=data_base_path)
# %% markdown
# ## Part 1
# In this part, we will study classification based on the data available in the Matlab file `classification.mat` that you will under `lab-03-data/part1`.
# There are 3 data sets in this file, each one being a training set for a given class.
# They are contained in variables `a`, `b` and `c`.
#
# **Note**: we can load Matlab files using the [scipy.io] module.
#
# [scipy.io]: https://docs.scipy.org/doc/scipy/reference/io.html
# %% codecell
import scipy.io

data_part1_path = os.path.join(data_base_path, data_folder, 'part1', 'classification.mat')
matfile = scipy.io.loadmat(data_part1_path)
a = matfile['a']
b = matfile['b']
c = matfile['c']

print(a.shape, b.shape, c.shape)
# %% markdown
# ### 1.1 Bayes method
# Using the Bayes method, give the analytical expression of the separation curves between those three classes.
# Do reasonable hypotheses about the distributions of those classes and estimate the corresponding parameters based on the given training sets.
# Draw those curves on a plot, together with the training data.
# For simplicity reasons, round the estimated parameters to the closest integer value.
#
# *Add your implementation and discussion*
# %% markdown
# ### 1.2 Mahalanobis distance
# For classes `a` and `b`, give the expression of the Mahalanobis distance used to classify a point in class `a` or `b`, and verify the obtained classification, in comparison with the "complete" Bayes classification, for a few points of the plane.
#
# *Add your implementation and discussion*
# %% markdown
# ## Part 2
# In this part, we aim to classify digits using the complete version of MNIST digits dataset.
# The dataset consists of 60'000 training images and 10'000 test images of handwritten digits.
# Each image has size 28x28, and has assigned a label from zero to nine, denoting the digits value.
# Given this data, your task is to construct a Multilayer Perceptron (MLP) for supervised training and classification and evaluate it on the test images.
#
# Download the MNIST dataset (all 4 files) from http://yann.lecun.com/exdb/mnist/ under `lab-03-data/part2`.
# You can then use the script provided below to extract and load training and testing images in Python.
#
# To create an MLP you are free to choose any library.
# In case you don't have any preferences, we encourage you to use the [scikit-learn] package; it is a simple, efficient and free tool for data analysis and machine learning.
# In this [link][sklearn-example], you can find a basic example to see how to create and train an MLP using [scikit-learn].
# Your network should have the following properties:
# * Input `x`: 784-dimensional (i.e. 784 visible units representing the flattened 28x28 pixel images).
# * 100 hidden units `h`.
# * 10 output units `y`, i.e. the labels, with a value close to one in the i-th class representing a high probability of the input representing the digit `i`.
#
# If you need additional examples you can borrow some code from image classification tutorials.
# However, we recommend that you construct a minimal version of the network on your own to gain better insights.
#
# [scikit-learn]: http://scikit-learn.org/stable/index.html
# [sklearn-example]: http://scikit-learn.org/stable/modules/neural_networks_supervised.html
# %% markdown
# ### 2.1 Dataset loading
# Here we first declare the methods `extract_data` and `extract_labels` so that we can reuse them later in the code.
# Then we extract both the data and corresponding labels, and plot randomly some images and corresponding labels of the training set.
# %% codecell
import gzip
import numpy as np
import matplotlib.pyplot as plt

def extract_data(filename, image_shape, image_number):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(np.prod(image_shape) * image_number)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(image_number, image_shape[0], image_shape[1])
    return data


def extract_labels(filename, image_number):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * image_number)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels
# %% codecell
image_shape = (28, 28)
train_set_size = 60000
test_set_size = 10000

data_part2_folder = os.path.join(data_base_path, data_folder, 'part2')

train_images_path = os.path.join(data_part2_folder, 'train-images-idx3-ubyte.gz')
train_labels_path = os.path.join(data_part2_folder, 'train-labels-idx1-ubyte.gz')
test_images_path = os.path.join(data_part2_folder, 't10k-images-idx3-ubyte.gz')
test_labels_path = os.path.join(data_part2_folder, 't10k-labels-idx1-ubyte.gz')

train_images = extract_data(train_images_path, image_shape, train_set_size)
test_images = extract_data(test_images_path, image_shape, test_set_size)
train_labels = extract_labels(train_labels_path, train_set_size)
test_labels = extract_labels(test_labels_path, test_set_size)
# %% codecell
prng = np.random.RandomState(seed=123456789)  # seed to always re-draw the same distribution
plt_ind = prng.randint(low=0, high=train_set_size, size=10)

fig, axes = plt.subplots(1, 10, figsize=(12, 3))
for ax, im, lb in zip(axes, train_images[plt_ind], train_labels[plt_ind]):
    ax.imshow(im, cmap='gray')
    ax.axis('off')
    ax.set_title(lb)
# %% markdown
# ### 2.2 MLP
# The difficult part about a MLP is it creation from scratch, implementing the
# activation functions, backpropagation and so forth. But if done correclty, once
# such a class is completed, it is easy to implement. This can be seen in the
# code below, wich uses the MLP class of scikit-learn. Here it is important to
# understand the data format that is fed to the NN to make it work correctly.
# Additionally it is useful to knoww how the different parameters can be changed.
# Looking at the default values of the classifier we see that the only hidden
# layer set by default has alread 100 neurons, wich is what is required for this
# exercise (so no modification is needed). The input layer is defined through
# dimention of the samples (28x28=784) and similarly the output layer is defined
# by the number of nuique labels.

# %% md
### FUNC DEF
# %% codecell
def arr2vec(arr):
    vec=np.reshape(arr, arr.size)
    return vec

def rearrange_img_stack(img_stack,z_stack=True):
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

# %% md
# Before using a NN one needs to train it. This is done by calling the code line
# `MLP.fit(X,y)`

# %% codecell
from sklearn.neural_network import MLPClassifier

#prepare data
X=rearrange_img_stack(train_images,z_stack=False)
y=train_labels
#define classifier object
MLP=MLPClassifier()
#train MLP
MLP.fit(X,y)

# %% md
# Now it is time to test the neural network's performances:

# %% codecell
X_test=rearrange_img_stack(test_images,z_stack=False)
y_test=test_labels
y_est=MLP.predict(X_test)

#compute percentage of correct labels
corr_pc=100*np.sum(y_est==y_test)/y_test.size
print("%.2f%s of the estimated labels are correct" %(corr_pc,'%'))
#plot 10 random images from test set with the estimated labels
plt_ind = prng.randint(low=0, high=test_set_size, size=10)

fig, axes = plt.subplots(1, 10, figsize=(12, 3))
for ax, im, lb in zip(axes, test_images[plt_ind], y_est[plt_ind]):
    ax.imshow(im, cmap='copper')
    ax.axis('off')
    ax.set_title(lb)
