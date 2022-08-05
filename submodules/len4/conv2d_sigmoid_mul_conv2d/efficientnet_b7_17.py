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
        self.conv2d85 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid17 = Sigmoid()
        self.conv2d86 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x266, x263):
        x267=self.conv2d85(x266)
        x268=self.sigmoid17(x267)
        x269=operator.mul(x268, x263)
        x270=self.conv2d86(x269)
        return x270

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x266 = torch.randn(torch.Size([batch_size, 20, 1, 1]))
x263 = torch.randn(torch.Size([batch_size, 480, 28, 28]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x266, x263)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
