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
        self.conv2d132 = Conv2d(888, 84, kernel_size=(1, 1), stride=(1, 1))
        self.relu103 = ReLU()
        self.conv2d133 = Conv2d(84, 888, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid25 = Sigmoid()
        self.conv2d134 = Conv2d(888, 888, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x418, x417):
        x419=self.conv2d132(x418)
        x420=self.relu103(x419)
        x421=self.conv2d133(x420)
        x422=self.sigmoid25(x421)
        x423=operator.mul(x422, x417)
        x424=self.conv2d134(x423)
        return x424

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x418 = torch.randn(torch.Size([batch_size, 888, 1, 1]))
x417 = torch.randn(torch.Size([batch_size, 888, 7, 7]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x418, x417)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
