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
        self.conv2d26 = Conv2d(208, 26, kernel_size=(1, 1), stride=(1, 1))
        self.relu19 = ReLU()
        self.conv2d27 = Conv2d(26, 208, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid4 = Sigmoid()

    def forward(self, x80, x79):
        x81=self.conv2d26(x80)
        x82=self.relu19(x81)
        x83=self.conv2d27(x82)
        x84=self.sigmoid4(x83)
        x85=operator.mul(x84, x79)
        return x85

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x80 = torch.randn(torch.Size([batch_size, 208, 1, 1]))
x79 = torch.randn(torch.Size([batch_size, 208, 14, 14]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x80, x79)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
