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
        self.dropout4 = Dropout(p=0.0, inplace=False)
        self.linear6 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout5 = Dropout(p=0.0, inplace=False)

    def forward(self, x82):
        x83=self.dropout4(x82)
        x84=self.linear6(x83)
        x85=self.dropout5(x84)
        return x85

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
