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

    def forward(self, x27, x33, x39, x43):
        x44=torch.cat([x27, x33, x39, x43], 1)
        return x44

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x27 = torch.randn(torch.Size([batch_size, 64, 28, 28]))
x33 = torch.randn(torch.Size([batch_size, 128, 28, 28]))
x39 = torch.randn(torch.Size([batch_size, 32, 28, 28]))
x43 = torch.randn(torch.Size([batch_size, 32, 28, 28]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x27, x33, x39, x43)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
