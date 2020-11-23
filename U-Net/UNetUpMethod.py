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
            #aplies a 2d transposed convolution operator on the tesnor
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
        # so can concat then in z-direction. 
        # loss in width and height due to rounding down sizes in maxpool
        x1 = F.pad(x1, [diffX//2, diffX - diffX //2,
                        diffY//2, diffY - diffY//2])
        
        x = torch.cat([x2, x1], dim =1)
        
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


### Constructing U-net ###

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
        print('D-layer1:', x1.shape) #prints tensor shape after 1st layer
        
        x2 = self.down1(x1) #second layer - is a down layer
        print('D-layer2:', x2.shape) 
        
        x3 = self.down2(x2) #3rd layer
        print('D-layer3:', x3.shape) 
        
        x4 = self.down3(x3) #4th layer
        print('D-layer4:', x4.shape)
        
        x5 = self.down4(x4) #5th layer
        print('D-layer5:', x5.shape)
        
        x = self.up1(x5,x4) #6th layer - is an Up layer
        print('U-layer1:', x.shape)
        
        x = self.up2(x,x3) #7th layer
        print('U-layer2:', x.shape)
        
        x = self.up3(x,x2) #8th layer
        print('U-layer3:', x.shape)
        
        x = self.up4(x,x1) #9th layer
        print('U-layer4:', x.shape)
        
        result = self.outc(x) #10th layer - is out layer
        
        return result #returns the resulting tensor

    
### Running U-net ###

U_net = Unet(1,1) #creates instance of the u-net 

t = torch.ones([1,1,572,572]) #placeholder 4D tensor of all ones 
# [batch_size, channels, height, width]
print('original shape:', t.shape)


out = U_net(t) #runs placeholder tensor through network
print('final:',out.shape) #prints shape of output tesnor to see if U-net has worked
