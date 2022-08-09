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
        self.layernorm28 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)

    def forward(self, x301, x315, x323):
        x316=operator.add(x301, x315)
        x324=operator.add(x316, x323)
        x325=self.layernorm28(x324)
        return x325

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x301 = torch.randn(torch.Size([batch_size, 7, 7, 768]))
