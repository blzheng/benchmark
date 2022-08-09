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
        self.dropout22 = Dropout(p=0.0, inplace=False)
        self.linear25 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout23 = Dropout(p=0.0, inplace=False)

    def forward(self, x304):
        x305=self.dropout22(x304)
        x306=self.linear25(x305)
        x307=self.dropout23(x306)
        return x307

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
