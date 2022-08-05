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
        self.conv2d31 = Conv2d(208, 52, kernel_size=(1, 1), stride=(1, 1))
        self.relu23 = ReLU()
        self.conv2d32 = Conv2d(52, 208, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid5 = Sigmoid()

    def forward(self, x96, x95):
        x97=self.conv2d31(x96)
        x98=self.relu23(x97)
        x99=self.conv2d32(x98)
        x100=self.sigmoid5(x99)
        x101=operator.mul(x100, x95)
        return x101

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x96 = torch.randn(torch.Size([batch_size, 208, 1, 1]))
x95 = torch.randn(torch.Size([batch_size, 208, 14, 14]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x96, x95)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
