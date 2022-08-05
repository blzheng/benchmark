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

    def forward(self, x51, x58, x65, x72, x79, x86, x93, x100):
        x101=torch.cat([x51, x58, x65, x72, x79, x86, x93, x100], 1)
        return x101

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x51 = torch.randn(torch.Size([batch_size, 128, 28, 28]))
x58 = torch.randn(torch.Size([batch_size, 32, 28, 28]))
x65 = torch.randn(torch.Size([batch_size, 32, 28, 28]))
x72 = torch.randn(torch.Size([batch_size, 32, 28, 28]))
x79 = torch.randn(torch.Size([batch_size, 32, 28, 28]))
x86 = torch.randn(torch.Size([batch_size, 32, 28, 28]))
x93 = torch.randn(torch.Size([batch_size, 32, 28, 28]))
x100 = torch.randn(torch.Size([batch_size, 32, 28, 28]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x51, x58, x65, x72, x79, x86, x93, x100)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
