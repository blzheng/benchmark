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
        self.relu55 = ReLU()
        self.conv2d73 = Conv2d(196, 784, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid13 = Sigmoid()
        self.conv2d74 = Conv2d(784, 784, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x227, x225):
        x228=self.relu55(x227)
        x229=self.conv2d73(x228)
        x230=self.sigmoid13(x229)
        x231=operator.mul(x230, x225)
        x232=self.conv2d74(x231)
        return x232

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x227 = torch.randn(torch.Size([batch_size, 196, 1, 1]))
x225 = torch.randn(torch.Size([batch_size, 784, 7, 7]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x227, x225)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
