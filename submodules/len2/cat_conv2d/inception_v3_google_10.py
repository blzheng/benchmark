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
        self.conv2d85 = Conv2d(2048, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x264, x274, x287, x291):
        x292=torch.cat([x264, x274, x287, x291], 1)
        x293=self.conv2d85(x292)
        return x293

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x264 = torch.randn(torch.Size([batch_size, 320, 5, 5]))
x274 = torch.randn(torch.Size([batch_size, 768, 5, 5]))
x287 = torch.randn(torch.Size([batch_size, 768, 5, 5]))
x291 = torch.randn(torch.Size([batch_size, 192, 5, 5]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x264, x274, x287, x291)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
