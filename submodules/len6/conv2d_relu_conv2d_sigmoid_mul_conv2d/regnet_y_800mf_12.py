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
        self.conv2d67 = Conv2d(784, 80, kernel_size=(1, 1), stride=(1, 1))
        self.relu51 = ReLU()
        self.conv2d68 = Conv2d(80, 784, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid12 = Sigmoid()
        self.conv2d69 = Conv2d(784, 784, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x210, x209):
        x211=self.conv2d67(x210)
        x212=self.relu51(x211)
        x213=self.conv2d68(x212)
        x214=self.sigmoid12(x213)
        x215=operator.mul(x214, x209)
        x216=self.conv2d69(x215)
        return x216

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x210 = torch.randn(torch.Size([batch_size, 784, 1, 1]))
x209 = torch.randn(torch.Size([batch_size, 784, 7, 7]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x210, x209)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
