import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

class ConvNet(nn.Module):
    def __init__(self, depth = 30, dropout=0.1):
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
        self.conv11 = nn.Conv2d(in_channels = 30, out_channels = int(depth/2), kernel_size = 1)
        # 12 = maxpool (output size = M x depth/2 x 3 x 47)

        self.length = int((depth/2) * 3 * 47)
        self.fc1 = nn.Linear(self.length, 60)
        self.fc2 = nn.Linear(60, 1)

        self.dout_layer = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.dout_layer(self.conv1(x)))
        x = F.relu(self.dout_layer(self.conv2(x)))
        x = self.maxpool(x)
        
        x = F.relu(self.dout_layer(self.conv4(x)))
        x = F.relu(self.dout_layer(self.conv5(x)))
        x = self.maxpool(x)

        x = F.relu(self.dout_layer(self.conv7(x)))
        x = F.relu(self.dout_layer(self.conv8(x)))
        x = self.maxpool(x)

        x = F.relu(self.dout_layer(self.conv10(x)))
        x = F.relu(self.dout_layer(self.conv11(x)))
        x = self.maxpool(x)

        x = x.view(-1, self.length)
        x = F.relu(self.dout_layer(self.fc1(x)))
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x
