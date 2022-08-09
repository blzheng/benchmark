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
        self.relu15 = ReLU()
        self.conv2d44 = Conv2d(168, 672, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid4 = Hardsigmoid()
        self.conv2d45 = Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x127, x125):
        x128=self.relu15(x127)
        x129=self.conv2d44(x128)
        x130=self.hardsigmoid4(x129)
        x131=operator.mul(x130, x125)
        x132=self.conv2d45(x131)
        return x132

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
