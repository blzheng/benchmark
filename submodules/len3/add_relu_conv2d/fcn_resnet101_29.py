import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator
import sys
import os

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.relu88 = ReLU(inplace=True)
        self.conv2d94 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x310, x302):
        x311=operator.add(x310, x302)
        x312=self.relu88(x311)
        x313=self.conv2d94(x312)
        return x313

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
