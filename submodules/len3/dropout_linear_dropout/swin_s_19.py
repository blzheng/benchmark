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
        self.dropout38 = Dropout(p=0.0, inplace=False)
        self.linear41 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout39 = Dropout(p=0.0, inplace=False)

    def forward(self, x488):
        x489=self.dropout38(x488)
        x490=self.linear41(x489)
        x491=self.dropout39(x490)
        return x491

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
