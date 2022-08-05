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
        self.conv2d8 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x31, x4, x11, x18, x25, x39):
        x32=self.conv2d8(x31)
        x40=torch.cat([x4, x11, x18, x25, x32, x39], 1)
        return x40

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x31 = torch.randn(torch.Size([batch_size, 128, 56, 56]))
x4 = torch.randn(torch.Size([batch_size, 64, 56, 56]))
x11 = torch.randn(torch.Size([batch_size, 32, 56, 56]))
x18 = torch.randn(torch.Size([batch_size, 32, 56, 56]))
x25 = torch.randn(torch.Size([batch_size, 32, 56, 56]))
x39 = torch.randn(torch.Size([batch_size, 32, 56, 56]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x31, x4, x11, x18, x25, x39)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
