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
        self.dropout20 = Dropout(p=0.0, inplace=False)
        self.linear23 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout21 = Dropout(p=0.0, inplace=False)

    def forward(self, x281):
        x282=self.dropout20(x281)
        x283=self.linear23(x282)
        x284=self.dropout21(x283)
        return x284

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
