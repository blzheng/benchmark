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
        self.dropout30 = Dropout(p=0.0, inplace=False)
        self.linear33 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout31 = Dropout(p=0.0, inplace=False)

    def forward(self, x396):
        x397=self.dropout30(x396)
        x398=self.linear33(x397)
        x399=self.dropout31(x398)
        return x399

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
