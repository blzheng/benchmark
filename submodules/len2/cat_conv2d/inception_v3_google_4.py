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
        self.conv2d30 = Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x102, x111, x112):
        x113=torch.cat([x102, x111, x112], 1)
        x114=self.conv2d30(x113)
        return x114

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x102 = torch.randn(torch.Size([batch_size, 384, 12, 12]))
x111 = torch.randn(torch.Size([batch_size, 96, 12, 12]))
x112 = torch.randn(torch.Size([batch_size, 288, 12, 12]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x102, x111, x112)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
