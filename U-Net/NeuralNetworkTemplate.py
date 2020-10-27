"""
Neural Network Template

@author: George Hume
"""
### IMPORTS ###

#PyTorch
import torch
torch.set_grad_enabled(True) #turns on/off computational graph needed for training
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#TorchVision
import torchvision
import torchvision.transforms as transforms 

#NumPy
import numpy as np


### DEFINING NETWORK ###
class Network(nn.Module):
    
    #Initaliser
    def __init__(self):
        super().__init__() #same attritbtues as super class (nn.Module)
            
    #Forward Pass Method
    def forward(self, t):
        return t

network = Network() #creates an instance of the network

t = torch.Tensor([[1,2,3,4],
                  [5,6,7,8],
                  [9,8,7,6],
                  [5,4,3,2]]) #placeholder tensor

result = network(t) #passes tensor through network

print(result)