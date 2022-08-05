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

    def forward(self, x256, x241, x272):
        x257=operator.add(x256, x241)
        x273=operator.add(x272, x257)
        return x273

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x256 = torch.randn(torch.Size([batch_size, 80, 28, 28]))
x241 = torch.randn(torch.Size([batch_size, 80, 28, 28]))
x272 = torch.randn(torch.Size([batch_size, 80, 28, 28]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x256, x241, x272)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
