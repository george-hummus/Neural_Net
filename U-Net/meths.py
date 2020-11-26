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
import torch.nn.functional as F

from PIL import Image
import numpy as np

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
        
## Up Method ##
class Up(nn.Module):
    '''Upscales the input tensor then performs double convolution on it to
    reduce its depth'''
    
    def __init__(self, in_channels, out_channels, bilinear = True):
        super().__init__() #imports attributes from super class (n.Module)
        
        #if bilinear is True then use normal convs to reduced tensor depth
        #we will only be using bilinear method
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
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
        
        #saftey measure that makes x1 have same height and width as x2
        # so can concat then in z-direction
        # shouldn't be needed if do padding in DuoCon
        x1 = F.pad(x1, [diffX//2, diffX - diffX //2,
                        diffY//2, diffY - diffY//2])
        
        x = torch.cat([x1, x2], dim =1)
        # Concatenates x1 and x2 in second dimension (along z-direction/depth)
        #therfore increases depth of tensor by 2
        
        return self.conv(x) 
        #does final double convolution on x to decrease the channel depth by 2
    

## Out Method ##
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__() #re-defining itself
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        #only method is to do a 2d convolution on the 
        #input tensor with a kernel size of 1
        
    def forward(self, x): #forward method of OutConv
        return self.conv(x) #does conv on tensor
  

## TIFF image loader and torch converter ##
def ImLoad(file_name):
    img = Image.open(file_name) #loads in TIFF file

    imgArray = np.zeros((img.size[1], img.size[0], img.n_frames), np.float32)
    #creates a balnk array to store TIFF frames in (32-bit float so compatible with pyTorch)
    for I in range(img.n_frames):
        img.seek(I)
        imgArray[:, :, I] = np.asarray(img) 
        #loops through TIFF frames and adds them to blank array
    img.close()
    
    imgArray = np.transpose(imgArray, (2,0,1)) 
    #changes array dimenisons so match that of tensors needed for pytorch
    batch, height, width = imgArray.shape[0],imgArray.shape[1],imgArray.shape[2]
    #assigns batch size, height and width of array to sepparte varibles
    imgArray = torch.as_tensor(imgArray) #turns array into tensor
    imgArray = imgArray.reshape(batch,1,height,width) #adds channel dimenion to tensor
    return(imgArray)

## Binary Step Activation fucntion ##
def BinStep(tensor):
    result = (tensor >0)
    result = torch.tensor(result, dtype=torch.uint8)
    return result
        