"""
DeepLizard Tutorial in making a neural network

@author: George Hume

"""

### IMPORTS ### 

import torch
#The top-level PyTorch package and tensor library. 
import torch.nn as nn
#A subpackage that contains modules and extensible classes for building
#neural networks. 
import torch.optim as optim
#A subpackage that contains standard optimization operations like SGD and Adam. 
import torch.nn.functional as F
#A functional interface that contains typical operations used for building
#neural networks like loss functions and convolutions. 

import torchvision
#A package that provides access to popular datasets, model architectures, 
#and image transformations for computer vision. 
import torchvision.transforms as transforms 
#An interface that contains common transforms for image processing. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
#from plotcm import plot_confusion_matrix
#above is a local file that'll be used later

import pdb #python debugger 

torch.set_printoptions(linewidth=120) 
#sets the print options for PyTorch print statements. 

###Creating instance of fashionMNIST data set###
train_set = torchvision.datasets.FashionMNIST(
    root='./data' #location where data is locateed
    ,train=True #the data set IS in the training set
    ,download=True #downloads the data
    ,transform=transforms.Compose([
        transforms.ToTensor() #converts the data set into a tensor
    ])
)
