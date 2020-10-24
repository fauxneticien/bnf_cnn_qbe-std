import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 30, kernel_size = 3)
        self.conv2 = nn.Conv2d(in_channels = 30, out_channels = 30, kernel_size = 3)
        self.maxpool  = nn.MaxPool2d(kernel_size = 2, stride= 2)

        self.conv4 = nn.Conv2d(in_channels = 30, out_channels = 30, kernel_size = 3)
        self.conv5 = nn.Conv2d(in_channels = 30, out_channels = 30, kernel_size = 3)
        # 6 = maxpool

        self.conv7 = nn.Conv2d(in_channels = 30, out_channels = 30, kernel_size = 3)
        self.conv8 = nn.Conv2d(in_channels = 30, out_channels = 30, kernel_size = 3)
        # 9 = maxpool

        self.conv10 = nn.Conv2d(in_channels = 30, out_channels = 30, kernel_size = 3)
        self.conv11 = nn.Conv2d(in_channels = 30, out_channels = 15, kernel_size = 3)
        # 12 = maxpool

        self.fc1 = nn.Linear(1380, 60)
        self.fc2 = nn.Linear(60, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.maxpool(x)

        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = self.maxpool(x)

        x = x.view(-1, 1380)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x
