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
        self.sigmoid53 = Sigmoid()
        self.conv2d266 = Conv2d(3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x835, x831):
        x836=self.sigmoid53(x835)
        x837=operator.mul(x836, x831)
        x838=self.conv2d266(x837)
        return x838

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x835 = torch.randn(torch.Size([batch_size, 3840, 1, 1]))
x831 = torch.randn(torch.Size([batch_size, 3840, 7, 7]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x835, x831)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
