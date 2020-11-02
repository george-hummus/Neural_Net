"""
Methods used to construct a U-Net.
    
    Code borrowed from:
        https://github.com/milesial/Pytorch-UNet/tree/master/unet

@author: George
"""


### IMPORTS ### 

import torch
torch.set_grad_enabled(True) 

import torch.nn as nn

torch.set_printoptions(linewidth=120) 

### Constructing the methods of the U-net ###

## Double Convolution Method ##
class DuoCon(nn.Module):
    '''This class is the Double Convolution Layer for the Down method.
    It will do a padded convolutuions on the input tensor, followed by a 
    batch normalisation and then a rectifer activation function. 
    This will be done twice for the 2 convolutions.
    '''
    
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super().__init__() #takes properties from nn.Model super class
        
        if not mid_channels:
            mid_channels = out_channels
            #gives option to have the output channels of the first convolution
            #to be different to the output channels of the last
            
        self.double_conv = nn.Sequential( 
            #creates a sequential list of intructions for the double_conv layer
            
            #first convolution
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            #filter is 3x3 and padding is 1 so whole tensor is sampled
            nn.BatchNorm2d(mid_channels),
            #normalises the channels out of the convolution
            nn.ReLU(inplace=True),
            #replaces all negative values with zero
            #implace argument means input is directly saved hence saving memeory
            
            #second convolution is same as first
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding =1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        
    def forward(self, x):
        return self.double_conv(x)
        #defining the forward method of the DuoCon class which is to just
        #run thru the double_conv layer
    
## Down Method ##
class Down(nn.Module):
    '''Downscales the output of the double convolution layer (with a maxpool
    method) and then does another double convolution on the output of this.
    '''
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), #halves the size of the input tensor
            DuoCon(in_channels, out_channels) #does another double conv
        )
        
    def forward(self, x):
        return self.maxpool_conv(x)
        #forward method is to run tensor through maxpooling 
        #and double conv layer