"""
U-Net class based of code found at:
    https://github.com/milesial/Pytorch-UNet/tree/master/unet

@author: George
"""
### IMPORTS ###
import torch
torch.set_grad_enabled(True) 

import torch.nn as nn
import torch.nn.functional as F

torch.set_printoptions(linewidth=120) 

#importing double convolution, Down layers, Up layers and Out layer from down method file
from meths import DuoCon
from meths import Down
from meths import Up
from meths import OutConv
from meths import BinStep #would be used as activation fucntion but can't 
#differetiate it so can't be use to find gradients in training loop


class Unet(nn.Module):
    '''
    Class defining the U-net by stating all its layers and defining its 
    forward pass mathod. 
    '''
    
    def __init__(self, n_channels, n_classes, bilinear=True):
        
        super(Unet, self).__init__() #takes attributes from super class
        
        factor = 2 
        #factor by which you want to decrease the no of channels in each up layer 
        
        ## defining atributes of u-net ##
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        ## defining the layers of the U-net ##
        self.first = DuoCon(n_channels, 64) 
        #double convoltion that outputs tensor with 64 channels
        
        self.down1 = Down(64,128)
        #1st down convolution that outputs tensor with 128 channels from the 
        #input tensor of 64 channels
        
        self.down2 = Down(128,256) #outputs 256 channel tensor
        
        self.down3 = Down(256,512) #outputs 512 channel tensor
        
        self.down4 = Down(512,1024//factor) #outputs 512 channel tensor
        #this is because up bilinear up method doesn't reduce channel size by 2
        #only becomes 1024 in concatenation
        
        self.up1 = Up(1024, 512//factor, bilinear) #outputs 256 channel tensor
        
        self.up2 = Up(512, 256//factor, bilinear) #outputs 128 channel tensor
        
        self.up3 = Up(256, 128//factor, bilinear) #outputs 64 channel tensor
        
        self.up4 = Up(128, 64, bilinear) #outputs 64 channel tensor
        
        self.outc = OutConv(64, n_classes)  #outputs 1 channel tensor
        
    
    ## defining the U-net's forward pass ##
    def forward(self, x):
        x1 = self.first(x) #first later is a single double convolution
        
        x2 = self.down1(x1) #second layer - is a down layer
        
        x3 = self.down2(x2) #3rd layer
        
        x4 = self.down3(x3) #4th layer
        
        x5 = self.down4(x4) #5th layer
        
        x = self.up1(x5,x4) #6th layer - is an Up layer
        
        x = self.up2(x,x3) #7th layer
        
        x = self.up3(x,x2) #8th layer
        
        x = self.up4(x,x1) #9th layer
        
        result = self.outc(x) #10th layer - is out layer
        
        result = F.relu(F.hardtanh(result,-0.1,0.1)) 
        #approximation of binary step activation fucntion
        
        return result #returns the resulting tensor