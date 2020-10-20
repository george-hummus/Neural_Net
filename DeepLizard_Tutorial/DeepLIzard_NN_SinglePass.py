"""
DeepLizard Tutorial in making a neural network

@author: George Hume

"""

### IMPORTS ### 

import torch
#The top-level PyTorch package and tensor library. 
torch.set_grad_enabled(False) #turns off computational graph for when not training

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
        ### implementing the forward pass ###
        
        #(1) input layer is the idenity function
        t=t #passes through input layer
        
        # (2) hidden conv layer (all layers except in & out are 'hidden')
        t = self.conv1(t) #tensor operated on by conv1 layer
        t = F.relu(t) #rectifer activation function
        #turns all -ve values to zero and keeps all =ve values the same
        t = F.max_pool2d(t, kernel_size=2, stride=2) #max pooling operation
        #reduces the number of pixels in the image by disregarding the 
        #lower pixel count in the filter and retaining the max
        #in this case the filter size is 2x2 and the stride size (no. of rows
        #ypu go down each time) is also zero, so it will half the image size
        
        # (3) hidden conv layer
        t = self.conv2(t) #tensor operated on by conv2 layer
        t = F.relu(t) #rectifer activation function
        t = F.max_pool2d(t, kernel_size=2, stride=2) #max pooling operation
        
        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4) #flattens tesnor so that it is 1D
        #length of tensor is 12*4*4 as output of con2 layer has 12 channels
        #and each channel has a size of 4x4
        t = self.fc1(t) #pass through fc1 linar layer
        t = F.relu(t) ##rectifer activation function
        
        # (5) hidden linear layer
        #don't need to flatten again as output of fc1 will be flat
        t = self.fc2(t) #pass through fc2 linar layer
        t = F.relu(t) #rectifer activation function
        
        # (6) output layer
        t = self.out(t) #passes thrugh output layer
        t = F.softmax(t, dim=1) #softmax returns a positive probability for 
        #each of the prediction classes, and the probabilities sum to 1. 
        #where the classes are from 0->9 and correspond to a different item of clothing.
        
        return t

network = Network() #creates an instance of the network

print(network) #will print the string representation of the network
#can use netowork.layer to get the string representation of a layer
#can used network.layer.weights so access the weights of a layer

### Passing through a Single Image ###

sample = next(iter(train_set)) #loads single image from training set
image, label = sample #seperates image an label

# Inserts an additional dimension that represents a batch of size 1
# as network only accepts batches as inputs
batch = image.unsqueeze(0)

pred = network(batch) #makes a prediction by passing image through network
# image shape needs to have shape (batch_size × in_channels × H × W)
print(pred) #gives tesnor of length 10 with probabilites for each of the classes
#predictions will be different with each call of the network as 
#weights are assigned randomly
print(torch.argmax(pred)) #gives most likely label
print(label) #actual class of image
