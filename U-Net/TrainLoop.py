"""
Traning Loop for U-net

@author: George
"""
### IMPORTS ###
import torch
torch.set_grad_enabled(True) 

import torch.optim as optim
import torch.nn.functional as F

from UNet import Unet
from meths import ImLoad
from meths import get_perc_correct

from torch.utils.tensorboard import SummaryWriter
torch.set_printoptions(linewidth=120)

import matplotlib.pyplot as plt

### Importing Network and data ###
net = Unet(1,1) #creates a bilinear instance of the U-net

images, maps = ImLoad('data/phase_train_images.tif'), ImLoad('data/epi_train_maps.tif')
#loads in training data and training maps corresponding to the

optimiser = optim.Adam(net.parameters(), lr=0.01)


### TensorBoard Implementation ###
tb = SummaryWriter() 
#creates summary writer instance call tb, so we can pass data to tesnorboard


### Training Unet ###
for epoch in range(20): #loops thru for 10 epochs
    
    #so far loop only trains with one image
    img = images[0].reshape([1,1,230,270]) #extracts first training image
    map1 = maps[0].reshape([1,1,230,270]) #extracts first map
    
    preds = net(img) #passes batch through network
    
    loss = F.binary_cross_entropy(preds,map1)
    #calculates the loss using the l1 loss fucntion
        
    optimiser.zero_grad()#zeros gradients before adding newly calculated ones
    
    loss.backward() #calculates gradients
    
    optimiser.step() # Updating the weights using the gradients
        
    total_loss = loss.item()
    perc_correct = get_perc_correct(preds, map1)
    #increases the total loss and no. correct for the epoch from each batch
    
    
    ## adding loss, number of correct preds, accuracy to tensorboard
    tb.add_scalar('Loss', total_loss, epoch)
    tb.add_scalar('Percentage Correct', perc_correct, epoch)
    
    #prints out the total loss and perc. of correct predictions at the end of epoch
    print(
          f'epoch: \t{epoch} \n',
          f'average percentage correct: \t{perc_correct} \n',
          f'loss: \t{total_loss} \n'
          '')

pred = preds.detach() #deatches final prediction so it can be plotted
plt.imshow(pred.reshape([230,270]), cmap='gray') #plots final prediction