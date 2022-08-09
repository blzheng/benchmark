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
        self.layernorm49 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear47 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu22 = GELU(approximate='none')
        self.dropout44 = Dropout(p=0.0, inplace=False)

    def forward(self, x554, x568):
        x569=operator.add(x554, x568)
        x570=self.layernorm49(x569)
        x571=self.linear47(x570)
        x572=self.gelu22(x571)
        x573=self.dropout44(x572)
        return x573

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x554 = torch.randn(torch.Size([batch_size, 7, 7, 768]))
