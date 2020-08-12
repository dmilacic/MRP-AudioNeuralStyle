import torch 
import torch.nn as nn


# CNN Model (3 conv layer)
    
###################################

class CNN(nn.Module):
    def __init__(self, lchannels, in_channels, kernelsize):
        super(CNN, self).__init__()
        
        self.lchannels = lchannels
        self.in_channels = in_channels      
        
        self.maxpool_kernel = (1, 2)
        self.conv_kernel = (1, kernelsize)
        self.padding_size = (0, (kernelsize-1)//2)
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, self.lchannels, kernel_size=self.conv_kernel, padding=self.padding_size),
            nn.BatchNorm2d(self.lchannels),
            nn.ReLU(),
            nn.MaxPool2d(self.maxpool_kernel))
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.lchannels, self.lchannels, kernel_size=self.conv_kernel, padding=self.padding_size),
            nn.BatchNorm2d(self.lchannels),
            nn.ReLU(),
            nn.MaxPool2d(self.maxpool_kernel))
        self.layer3 = nn.Sequential(
            nn.Conv2d(self.lchannels, self.lchannels, kernel_size=self.conv_kernel, padding=self.padding_size),
            nn.BatchNorm2d(self.lchannels),
            nn.ReLU(),
            nn.MaxPool2d(self.maxpool_kernel))
        
    def forward(self, x):
        out = self.layer1(x)
        return out