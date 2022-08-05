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
        self.conv2d13 = Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1))
        self.relu13 = ReLU(inplace=True)
        self.conv2d14 = Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
        self.relu14 = ReLU(inplace=True)

    def forward(self, x32):
        x33=self.conv2d13(x32)
        x34=self.relu13(x33)
        x35=self.conv2d14(x34)
        x36=self.relu14(x35)
        return x36

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x32 = torch.randn(torch.Size([batch_size, 256, 27, 27]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x32)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
