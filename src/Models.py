import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

# Architecture taken from original code here: https://github.com/idiap/CNN_QbE_STD/blob/master/Model_Query_Detection_DTW_CNN.py

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

class VGG(nn.Module):
    def __init__(self, vgg_name):
        cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }

        def _make_layers(cfg, kernel=3):
            layers = []
            in_channels = 1
            for x in cfg:
                if x == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=kernel, padding=1),
                                nn.BatchNorm2d(x),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p = 0.1)]
                    in_channels = x
            layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
            return nn.Sequential(*layers)

        super(VGG, self).__init__()
        self.features = _make_layers(cfg[vgg_name])
        self.fc1 = nn.Linear(38400, 512)
        self.fc2 = nn.Linear(512, 1)
        self.dout_layer = nn.Dropout(0.1)

    def forward(self, x):
        # out = self.features(x)
        for m in self.features.children():
            # x_in = x.shape
            x = m(x)
            # print("%s -> %s" % (x_in, x.shape))

        out = x
        out = out.view(out.size(0), -1)
        out = self.dout_layer(self.fc1(out))
        out = self.fc2(out)
        return torch.sigmoid(out)

class VGG11(VGG):
    def __init__(self):
        VGG.__init__(self, 'VGG11')
