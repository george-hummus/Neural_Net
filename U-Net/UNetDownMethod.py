"""
U-Net Down Method
    Code to implemnt the down process (encoder section) of a U-Net
    Doubly convoutes tensors then max pools them
    
    Code borrowed from:
        https://github.com/milesial/Pytorch-UNet/tree/master/unet

@author: George
"""
### IMPORTS ### 

import torch
torch.set_grad_enabled(True) 

import torch.nn as nn

torch.set_printoptions(linewidth=120) 

### Constructing the layers of the Down Method ###

## Double Convolution Layer ##
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
    
##Down Layer##

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
            
            
### Constructing U-net ###

class Unet(nn.Module):
    '''
    Class defining the U-net by stating all its layers and defining its 
    forward pass mathod. 
    '''
    
    def __init__(self, n_channels, n_classes):
        
        super(Unet, self).__init__() #takes attributes from super class
        
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
        
        self.down4 = Down(512,1024) #outputs 1024 channel tensor
        
    
    ## defining the U-net's forward pass ##
    def forward(self, x):
        x1 = self.first(x) #first later is a single double convolution
        print('layer1:', x1.shape) #prints tensor shape after 1st layer
        
        x2 = self.down1(x1) #second layer is a down layer
        print('layer2:', x2.shape) 
        
        x3 = self.down2(x2) #3rd layer
        print('layer3:', x3.shape) 
        
        x4 = self.down3(x3) #4th layer
        print('layer4:', x4.shape)
        
        x5 = self.down4(x4) #5th layer
        result = x5
        return result #returns the resulting tensor
    
### Running U-net ###

U_net = Unet(1,1) #creates instance of the u-net 

t = torch.ones([1,1,572,572]) #placeholder 4D tensor of all ones 
# [batch_size, channels, height, width]
print('original shape:', t.shape)

out = U_net(t) #runs placeholder tensor through network
print('final:',out.shape) #prints shape of output tesnor to see if U-net has worked