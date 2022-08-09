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
        self.layernorm27 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear25 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu11 = GELU(approximate='none')
        self.dropout22 = Dropout(p=0.0, inplace=False)
        self.linear26 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout23 = Dropout(p=0.0, inplace=False)

    def forward(self, x301, x315):
        x316=operator.add(x301, x315)
        x317=self.layernorm27(x316)
        x318=self.linear25(x317)
        x319=self.gelu11(x318)
        x320=self.dropout22(x319)
        x321=self.linear26(x320)
        x322=self.dropout23(x321)
        return x322

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x301 = torch.randn(torch.Size([batch_size, 7, 7, 768]))
