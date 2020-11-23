"""
DeepLizard Tutorial in making a neural network.
    Wokring with dataset object

@author: George Hume

"""

### IMPORTS ### 

import torch
#The top-level PyTorch package and tensor library. 

import torchvision
#A package that provides access to popular datasets, model architectures, 
#and image transformations for computer vision. 
import torchvision.transforms as transforms 
#An interface that contains common transforms for image processing. 

import matplotlib.pyplot as plt



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

print(train_set.targets.bincount()) 
#number of samples in each class are equal so data set is balanced

sample = next(iter(train_set))#iter converts train_set into a stream of data
#next function gets next element in data stream

print(len(sample)) 
#length of element is two, as it contains image and label (both are tensors)

image = sample[0].squeeze()
#extraxts image from sample and squuezes to remove colour channel axis 

#display image
plt.imshow(image, cmap="gray")

im = image.numpy()




