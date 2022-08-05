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
        self.conv2d1 = Conv2d(96, 16, kernel_size=(1, 1), stride=(1, 1))
        self.relu1 = ReLU(inplace=True)
        self.conv2d2 = Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
        self.relu2 = ReLU(inplace=True)

    def forward(self, x3):
        x4=self.conv2d1(x3)
        x5=self.relu1(x4)
        x6=self.conv2d2(x5)
        x7=self.relu2(x6)
        return x7

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x3 = torch.randn(torch.Size([batch_size, 96, 54, 54]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x3)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
