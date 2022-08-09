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
        self.dropout34 = Dropout(p=0.0, inplace=False)
        self.linear37 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout35 = Dropout(p=0.0, inplace=False)

    def forward(self, x442):
        x443=self.dropout34(x442)
        x444=self.linear37(x443)
        x445=self.dropout35(x444)
        return x445

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
