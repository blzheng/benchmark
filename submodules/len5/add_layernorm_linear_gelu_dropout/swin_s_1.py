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
        self.layernorm4 = LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        self.linear2 = Linear(in_features=96, out_features=384, bias=True)
        self.gelu1 = GELU(approximate='none')
        self.dropout2 = Dropout(p=0.0, inplace=False)

    def forward(self, x26, x40):
        x41=operator.add(x26, x40)
        x42=self.layernorm4(x41)
        x43=self.linear2(x42)
        x44=self.gelu1(x43)
        x45=self.dropout2(x44)
        return x45

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x26 = torch.randn(torch.Size([batch_size, 56, 56, 96]))
x40 = torch.randn(torch.Size([batch_size, 56, 56, 96]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x26, x40)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
