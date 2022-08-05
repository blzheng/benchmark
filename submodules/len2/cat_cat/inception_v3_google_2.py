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

    def forward(self, x301, x304, x295, x318, x322):
        x305=torch.cat([x301, x304], 1)
        x323=torch.cat([x295, x305, x318, x322], 1)
        return x323

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x301 = torch.randn(torch.Size([batch_size, 384, 5, 5]))
x304 = torch.randn(torch.Size([batch_size, 384, 5, 5]))
x295 = torch.randn(torch.Size([batch_size, 320, 5, 5]))
x318 = torch.randn(torch.Size([batch_size, 768, 5, 5]))
x322 = torch.randn(torch.Size([batch_size, 192, 5, 5]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x301, x304, x295, x318, x322)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
