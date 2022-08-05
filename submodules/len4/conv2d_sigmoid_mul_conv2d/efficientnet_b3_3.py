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
        self.conv2d17 = Conv2d(8, 192, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid3 = Sigmoid()
        self.conv2d18 = Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x50, x47):
        x51=self.conv2d17(x50)
        x52=self.sigmoid3(x51)
        x53=operator.mul(x52, x47)
        x54=self.conv2d18(x53)
        return x54

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x50 = torch.randn(torch.Size([batch_size, 8, 1, 1]))
x47 = torch.randn(torch.Size([batch_size, 192, 56, 56]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x50, x47)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
