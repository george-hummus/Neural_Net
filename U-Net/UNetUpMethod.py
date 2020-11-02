"""
U-Net Up Method
    Code to implemnt the down process (encoder section) and Up process 
    (decoder section) of a U-Net.
    
    Code borrowed from:
        https://github.com/milesial/Pytorch-UNet/tree/master/unet

@author: George
"""

### IMPORTS ### 

import torch
torch.set_grad_enabled(True) 

import torch.nn as nn

import torch.nn.functional as F

torch.set_printoptions(linewidth=120) 

#importing double convolution and Down layers from down method file
from meths import DuoCon 
from meths import Down

### Constructing the layers of the Up Method ###

## Up Method ##
class Up(nn.Module):
    '''Upscales the input tensor then performs double convolution on it to
    reduce its depth'''
    
    def __init__(self, in_channels, out_channels, bilinear = True):
        super().__init__() #imports attributes from super class (n.Module)
        
        #if bilinear is True then use normal convs to reduced tensor depth
        if bilinear: #this is method used in U-net diagram?
            self.up = nn.Upsample(scale_factor=2, mode= bilinear, align_corners=True)
            #does a bilinear upscaling of the tensor and increase its width and height by 2.
            #bilinear upsampling uses nearby pixels to calculate new pixel values 
            #when upscaling, using linear interpolation.
            #align_corners being true means values in corner pixels are preserved.
            
            self.conv = DuoCon(in_channels, out_channels, in_channels//2)
            #double convolution where in first convolution channels are decreased by 2
            #in second channels are decreases so they match the out_channels
            
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride = 2)
            #aplies a 2d trnasposed convolution operator on the tesnor
            #this increases its size by 2 and reduces its depth by a factor of 2
            self.conv = DuoCon(in_channels,out_channels)
            #double convolution gets tensor to depth required by outchannels
            
    def forward(self,x1,x2): #forward method of Up layer
        #input is x1 which is tensor inputted into layer
        #x2 is tensor from opposite side of U-net that is concencrated with x1
        
        x1 = self.up(x1) #performs up method on input tensor x1
        
        #finds difference between x1 height and width
        diffY = x2.size()[2] - x1.size()[2] 
        # index 2 os height as 0 and 1 are batch and channels
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX//2, diffX - diffX //2,
                        diffY//2, diffY - diffY//2])
        #pads last 2 dimensions of x1 in  so that it has the same size as x2
        
        x = torch.cat([x1, x2], dim =1)
        # Concatenates x1 and x2 in second dimension (along width)
        #therfore increases width of tensor by 2
        
        return self.conv(x) #does final double convolution on x and returns it
    

## Out Method ##
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__() #?
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        #only method is to do a 2d convolution on the 
        #input tensor with a kernel size of 1
        
    def forward(self, x): #forward method of OutConv
        return self.conv(x) #does conv on tensor