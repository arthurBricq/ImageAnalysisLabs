
import os
import scipy.io
import matplotlib.pyplot as plt 



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
        
    
