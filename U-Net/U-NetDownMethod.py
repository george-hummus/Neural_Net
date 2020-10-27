"""
U-Net Down Method
    Code to implemnt the down process (encoder section) of a U-Net
    Doubly convoutes tensors then max pools them

@author: George
"""
### IMPORTS ### 

import torch
torch.set_grad_enabled(True) 

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms 

from torch.utils.tensorboard import SummaryWriter

torch.set_printoptions(linewidth=120) 