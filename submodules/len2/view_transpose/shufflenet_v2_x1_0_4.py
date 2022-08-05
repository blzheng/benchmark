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

    def forward(self, x108, x110, x114, x112, x113):
        x115=x108.view(x110, 2, x114, x112, x113)
        x116=torch.transpose(x115, 1, 2)
        return x116

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x108 = torch.randn(torch.Size([batch_size, 232, 14, 14]))
x110 = batch_size
x114 = batch_size16
x112 = batch_size4
x113 = batch_size4
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x108, x110, x114, x112, x113)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
