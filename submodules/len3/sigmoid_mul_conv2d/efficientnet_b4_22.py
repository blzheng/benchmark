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
        self.sigmoid22 = Sigmoid()
        self.conv2d113 = Conv2d(960, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x349, x345):
        x350=self.sigmoid22(x349)
        x351=operator.mul(x350, x345)
        x352=self.conv2d113(x351)
        return x352

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x349 = torch.randn(torch.Size([batch_size, 960, 1, 1]))
x345 = torch.randn(torch.Size([batch_size, 960, 7, 7]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x349, x345)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
