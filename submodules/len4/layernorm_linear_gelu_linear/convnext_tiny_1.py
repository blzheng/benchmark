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
        self.layernorm1 = LayerNorm((96,), eps=1e-06, elementwise_affine=True)
        self.linear2 = Linear(in_features=96, out_features=384, bias=True)
        self.gelu1 = GELU(approximate=none)
        self.linear3 = Linear(in_features=384, out_features=96, bias=True)

    def forward(self, x20):
        x21=self.layernorm1(x20)
        x22=self.linear2(x21)
        x23=self.gelu1(x22)
        x24=self.linear3(x23)
        return x24

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x20 = torch.randn(torch.Size([batch_size, 56, 56, 96]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x20)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
