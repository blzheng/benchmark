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
        self.conv2d55 = Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid11 = Sigmoid()
        self.conv2d56 = Conv2d(288, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x172, x169):
        x173=self.conv2d55(x172)
        x174=self.sigmoid11(x173)
        x175=operator.mul(x174, x169)
        x176=self.conv2d56(x175)
        return x176

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x172 = torch.randn(torch.Size([batch_size, 12, 1, 1]))
x169 = torch.randn(torch.Size([batch_size, 288, 28, 28]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x172, x169)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
