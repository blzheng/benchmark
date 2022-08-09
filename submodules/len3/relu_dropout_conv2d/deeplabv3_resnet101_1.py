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
        self.relu107 = ReLU()
        self.dropout1 = Dropout(p=0.1, inplace=False)
        self.conv2d113 = Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x375):
        x376=self.relu107(x375)
        x377=self.dropout1(x376)
        x378=self.conv2d113(x377)
        return x378

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
