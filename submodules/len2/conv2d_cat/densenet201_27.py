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
        self.conv2d62 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x223, x140, x147, x154, x161, x168, x175, x182, x189, x196, x203, x210, x217, x231):
        x224=self.conv2d62(x223)
        x232=torch.cat([x140, x147, x154, x161, x168, x175, x182, x189, x196, x203, x210, x217, x224, x231], 1)
        return x232

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x223 = torch.randn(torch.Size([batch_size, 128, 14, 14]))
x140 = torch.randn(torch.Size([batch_size, 256, 14, 14]))
x147 = torch.randn(torch.Size([batch_size, 32, 14, 14]))
x154 = torch.randn(torch.Size([batch_size, 32, 14, 14]))
x161 = torch.randn(torch.Size([batch_size, 32, 14, 14]))
x168 = torch.randn(torch.Size([batch_size, 32, 14, 14]))
x175 = torch.randn(torch.Size([batch_size, 32, 14, 14]))
x182 = torch.randn(torch.Size([batch_size, 32, 14, 14]))
x189 = torch.randn(torch.Size([batch_size, 32, 14, 14]))
x196 = torch.randn(torch.Size([batch_size, 32, 14, 14]))
x203 = torch.randn(torch.Size([batch_size, 32, 14, 14]))
x210 = torch.randn(torch.Size([batch_size, 32, 14, 14]))
x217 = torch.randn(torch.Size([batch_size, 32, 14, 14]))
x231 = torch.randn(torch.Size([batch_size, 32, 14, 14]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x223, x140, x147, x154, x161, x168, x175, x182, x189, x196, x203, x210, x217, x231)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
