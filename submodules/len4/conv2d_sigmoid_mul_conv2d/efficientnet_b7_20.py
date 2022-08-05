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
        self.conv2d100 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid20 = Sigmoid()
        self.conv2d101 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x312, x309):
        x313=self.conv2d100(x312)
        x314=self.sigmoid20(x313)
        x315=operator.mul(x314, x309)
        x316=self.conv2d101(x315)
        return x316

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x312 = torch.randn(torch.Size([batch_size, 40, 1, 1]))
x309 = torch.randn(torch.Size([batch_size, 960, 14, 14]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x312, x309)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
