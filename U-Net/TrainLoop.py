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
from meths import get_perc_correct

import matplotlib.pyplot as plt

### Importing Network and data ###
net = Unet(1,1) #creates a bilinear instance of the U-net

images, maps = ImLoad('data/phase_train_images.tif'), ImLoad('data/epi_train_maps.tif')
#loads in training data and training maps corresponding to the

optimiser = optim.Adam(net.parameters(), lr=0.01)


### Training Unet ###
for epoch in range(10): #loops thru for 10 epochs
    
    total_loss=0
    perc_correct=0
    
    img = images[0].reshape([1,1,230,270])
    map1 = maps[0].reshape([1,1,230,270])
    
    preds = net(img) #passes batch through network
    
    loss = F.binary_cross_entropy(preds,map1)
    #calculates the loss using the l1 loss fucntion
        
    optimiser.zero_grad()#zeros gradients before adding newly calculated ones
    
    loss.backward() #calculates gradients
    
    optimiser.step() # Updating the weights using the gradients
        
    total_loss += loss.item()
    perc_correct += get_perc_correct(preds, map1)
    #increases the total loss and no. correct for the epoch from each batch
    
    #prints out the total loss and no. of correct predictions at the 
    #end of the above loop
    print(
          'epoch:', epoch,
          'average perc correct:', perc_correct,
          'loss:', total_loss
          )
    
    pred = preds.detach() #deatches final prediction so it can be plotted
    plt.imshow(pred.reshape([230,270]), cmap='gray') #plots final prediction