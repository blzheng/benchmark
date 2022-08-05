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
        self.maxpool2d2 = MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv2d13 = Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x29, x31):
        x32=torch.cat([x29, x31], 1)
        x33=self.maxpool2d2(x32)
        x34=self.conv2d13(x33)
        return x34

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x29 = torch.randn(torch.Size([batch_size, 128, 27, 27]))
x31 = torch.randn(torch.Size([batch_size, 128, 27, 27]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x29, x31)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
