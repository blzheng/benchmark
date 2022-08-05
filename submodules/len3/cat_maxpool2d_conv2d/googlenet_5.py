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
        self.maxpool2d12 = MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.conv2d56 = Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x169, x175, x181, x185):
        x186=torch.cat([x169, x175, x181, x185], 1)
        x202=self.maxpool2d12(x186)
        x203=self.conv2d56(x202)
        return x203

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x169 = torch.randn(torch.Size([batch_size, 256, 7, 7]))
x175 = torch.randn(torch.Size([batch_size, 320, 7, 7]))
x181 = torch.randn(torch.Size([batch_size, 128, 7, 7]))
x185 = torch.randn(torch.Size([batch_size, 128, 7, 7]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x169, x175, x181, x185)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
