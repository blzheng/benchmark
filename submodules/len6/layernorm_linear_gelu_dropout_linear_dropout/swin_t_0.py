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
        self.layernorm2 = LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        self.linear0 = Linear(in_features=96, out_features=384, bias=True)
        self.gelu0 = GELU(approximate='none')
        self.dropout0 = Dropout(p=0.0, inplace=False)
        self.linear1 = Linear(in_features=384, out_features=96, bias=True)
        self.dropout1 = Dropout(p=0.0, inplace=False)

    def forward(self, x18):
        x19=self.layernorm2(x18)
        x20=self.linear0(x19)
        x21=self.gelu0(x20)
        x22=self.dropout0(x21)
        x23=self.linear1(x22)
        x24=self.dropout1(x23)
        return x24

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x18 = torch.randn(torch.Size([batch_size, 56, 56, 96]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x18)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
