"""
DeepLizard Tutorial in making a neural network.
    Wokring with DataLoader object, which eventually will be used within
    the neural network training loop.

@author: George
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

###Creating instance of fashionMNIST data set###
train_set = torchvision.datasets.FashionMNIST(
    root='./data' #location where data is locateed
    ,train=True #the data set IS in the training set
    ,download=True #downloads the data
    ,transform=transforms.Compose([
        transforms.ToTensor() #converts the data set into a tensor
    ])
)

#creating a smaller data loader with a bacth size of 10
display_loader = torch.utils.data.DataLoader(
    train_set, batch_size=10
) 
#if shuffle = True then each batch will be different for each call
#(shuffle = False by deafault)

batch = next(iter(display_loader))#iter converts train_set into a stream of data
#next function gets next element in data stream

print('len of batch:', len(batch))

images, labels = batch #sepparetes batch into images and labels

print('types:', type(images), type(labels)) #both tensors
print('shapes:', images.shape, labels.shape) 
#an extra axis on both labels and images will be 10 due to batch size

#size of images tesnor given by: (
#batch size, no. of colour channels, image height, image width)

##plotting a batch of images##

grid = torchvision.utils.make_grid(images, nrow=10) 
#torch vison fucntion to create a grid of images

plt.figure(figsize=(15,15))
plt.imshow(grid.permute(1,2,0)) #plots grid

print('labels:', labels) 
#prints the labels of the corresponding types of clothing



