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
        self.layernorm51 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear49 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu23 = GELU(approximate='none')

    def forward(self, x592):
        x593=self.layernorm51(x592)
        x594=self.linear49(x593)
        x595=self.gelu23(x594)
        return x595

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
