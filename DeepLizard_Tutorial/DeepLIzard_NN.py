"""
DeepLizard Tutorial in making a neural network

@author: George Hume

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
#sets the print options for PyTorch print statements. 

###Creating instance of fashionMNIST data set###
train_set = torchvision.datasets.FashionMNIST(
    root='./data' #location where data is locateed
    ,train=True #the data set IS in the training set
    ,download=True #downloads the data
    ,transform=transforms.Compose([
        transforms.ToTensor() #converts the data set into a tensor
    ])
)

train_loader = torch.utils.data.DataLoader(train_set 
#loads data we have just created an instance of
    ,batch_size=100 #gets a batch of 100
    ,shuffle=True #shuffles the data
)

### Creating Neural Network Class ###

class Network(nn.Module): 
    #class is an extension of the nn.module class so inherits its methods
    def __init__(self):
        super().__init__() #same attritbtues as super class (nn.Module)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        #first convolution layer (in channels equals colour channels of images)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        #second convolution layer (in channels matches out channels of first)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        #first linear layer
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        #second linear layer
        self.out = nn.Linear(in_features=60, out_features=10)
        #final linear layer/ output layer

    def forward(self, t):
        ### implementing the forward pass ###
        
        #(1) input layer is the idenity function
        t=t #passes through input layer
        
        # (2) hidden conv layer (all layers except in & out are 'hidden')
        t = self.conv1(t) #tensor operated on by conv1 layer
        t = F.relu(t) #rectifer activation function
        t = F.max_pool2d(t, kernel_size=2, stride=2) #max pooling operation
        
        
        # (3) hidden conv layer
        t = self.conv2(t) #tensor operated on by conv2 layer
        t = F.relu(t) 
        t = F.max_pool2d(t, kernel_size=2, stride=2) 
        
        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4) #flattens tesnor so that it is 1D
        t = self.fc1(t) #pass through fc1 linar layer
        t = F.relu(t) ##rectifer activation function
        
        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)
        
        # (6) output layer
        t = self.out(t) #passes thrugh output layer
        
        return t

network = Network() #creates an instance of the network

#creating an optimiser to update the weights
optimiser = optim.Adam(network.parameters(), lr=0.01)

def get_num_correct(p, lbs): #finds the number ofpredictions that are correct
    return p.argmax(dim=1).eq(lbs).sum().item()

### TensorBoard Implementation ###
images, labels = next(iter(train_loader)) 
#extracts images and labels so we can use them in tensorboard
grid = torchvision.utils.make_grid(images) #turns image batch into gird of images

tb = SummaryWriter() 
#creates summary writer instance call tb, so we can pass data to tesnorboard
tb.add_image('images', grid) 
tb.add_graph(network, images) 

### Calculating the Loss ###
for epoch in range(10): #loops thru for 10 epochs
    
    total_loss=0
    total_correct=0
    
    for batch in train_loader: #gets a batch a loops through for each in train_loader
    
        images, labels = batch #sepparetes batch into images and labels
        
        preds = network(images) #passes batch through network
        loss = F.cross_entropy(preds,labels) 
        #calculates the loss using the cross entropy function
        
        optimiser.zero_grad()#zeros gradients before adding newly calculated ones
    
        loss.backward() #calculates gradients
    
        optimiser.step() # Updating the weights using the gradients
        
        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)
        #increases the total loss and no. correct for the epoch from each batch
    
    ## adding loss, number of correct preds, accuracy to tensorboard
    tb.add_scalar('Loss', total_loss, epoch)
    tb.add_scalar('Number Correct', total_correct, epoch)
    tb.add_scalar('Accuracy', total_correct / len(train_set), epoch)
    
    ## adds histograms tp tensorboard
    tb.add_histogram('conv1.bias', network.conv1.bias, epoch)
    tb.add_histogram('conv1.weight', network.conv1.weight, epoch)
    tb.add_histogram(
        'conv1.weight.grad'
        ,network.conv1.weight.grad
        ,epoch
    )
    
    #prints out the total loss and no. of correct predictions at the 
    #end of the above loop
    print(
          'epoch:', epoch,
          'total_correct:', total_correct,
          'loss:', total_loss
          )
