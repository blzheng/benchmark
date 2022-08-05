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
        self.conv2d3 = Conv2d(12, 48, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid0 = Sigmoid()
        self.conv2d4 = Conv2d(48, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x9, x6):
        x10=self.conv2d3(x9)
        x11=self.sigmoid0(x10)
        x12=operator.mul(x11, x6)
        x13=self.conv2d4(x12)
        return x13

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x9 = torch.randn(torch.Size([batch_size, 12, 1, 1]))
x6 = torch.randn(torch.Size([batch_size, 48, 112, 112]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x9, x6)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
