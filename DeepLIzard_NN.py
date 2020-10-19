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

train_loader = torch.utils.data.DataLoader(train_set 
#loads data we have just created an instance of
    ,batch_size=1000 #gets a batch of 1000
    ,shuffle=True #shuffles the data
)

### Creating Neural Network Class ###

class Network(nn.Module): 
    #class is an extension of the nn.module class so inherits its methods
    def __init__(self):
        super().__init__() #same attritbtues as super class (nn.Module)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        #first convolution layer (in channels equals colour channels of images)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        #second convolution layer (in channels matches out channels of first)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        #first linear layer
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        #second linear layer
        self.out = nn.Linear(in_features=60, out_features=10)
        #final linear layer/ output layer

    def forward(self, t):
        # implement the forward pass
        return t

network = Network() #creates an instance of the network

print(network) #will print the string representation of the network
#can use netowork.layer to get the string representation of a layer
#can used network.layer.weights so access the weights of a layer


