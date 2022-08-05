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
        self.linear0 = Linear(in_features=128, out_features=512, bias=True)
        self.gelu0 = GELU(approximate=none)
        self.linear1 = Linear(in_features=512, out_features=128, bias=True)

    def forward(self, x10):
        x11=self.linear0(x10)
        x12=self.gelu0(x11)
        x13=self.linear1(x12)
        return x13

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x10 = torch.randn(torch.Size([batch_size, 56, 56, 128]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x10)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
