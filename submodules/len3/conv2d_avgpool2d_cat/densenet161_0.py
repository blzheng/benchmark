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
        self.conv2d13 = Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.avgpool2d0 = AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x49, x58):
        x50=self.conv2d13(x49)
        x51=self.avgpool2d0(x50)
        x59=torch.cat([x51, x58], 1)
        return x59

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x49 = torch.randn(torch.Size([batch_size, 384, 56, 56]))
x58 = torch.randn(torch.Size([batch_size, 48, 28, 28]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x49, x58)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
