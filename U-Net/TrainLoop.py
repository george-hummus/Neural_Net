"""
Traning Loop for U-net

@author: George
"""
### IMPORTS ###
import torch
torch.set_grad_enabled(True) 

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from UNet import Unet
from meths import ImLoad

import matplotlib.pyplot as plt
import numpy as np

### Importing Network and data ###
net = Unet(1,1) #creates a bilinear instance of the U-net

images, maps = ImLoad('data/train_images.tif'), ImLoad('data/train_maps.tif')
#loads in training data and training maps corresponding to the

optimiser = optim.Adam(net.parameters(), lr=0.01)

### Number correct function ###

def get_num_correct(Preds, Maps):
    
    percs = [] #empty list to fill with percentages
    for i in range(Preds.shape[0]): #loops through every image in batch
        p,m = Preds[i,0,:,:],  Maps[i,0,:,:] #extracts first prediction and map
        bp = (p>0.5) #converts predcition into boolean (all values lower that 0.5 false, higher true)
        bm = (m == 1) #converts map into boolean (true/flase)
        
        mask = (bp == bm)
        #creates mask comapring if predictions and maps
        num_correct = mask.sum().item()
        #sums mask to find number of pixels that were correct
        perc = (num_correct/(p.shape[0]*p.shape[1]))*100
        #finds percentage of all pixels that where correct
        percs.append(perc)
        #adds percentage to list  
    return np.mean(percs)


### Training Unet ###
for epoch in range(10): #loops thru for 10 epochs
    
    total_loss=0
    perc_correct=0
    
    img = images[0].reshape([1,1,230,270])
    
    preds = net(img) #passes batch through network
    
    loss = F.binary_cross_entropy(preds,maps[0])
    #calculates the loss using the cross entropy function
        
    optimiser.zero_grad()#zeros gradients before adding newly calculated ones
    
    loss.backward() #calculates gradients
    
    optimiser.step() # Updating the weights using the gradients
        
    total_loss += loss.item()
    perc_correct += get_num_correct(preds[0], maps[0])
    #increases the total loss and no. correct for the epoch from each batch
    
    #prints out the total loss and no. of correct predictions at the 
    #end of the above loop
    print(
          'epoch:', epoch,
          'total_correct:', perc_correct,
          'loss:', total_loss
          )