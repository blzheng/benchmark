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
        self.conv2d181 = Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid36 = Sigmoid()

    def forward(self, x567, x564):
        x568=self.conv2d181(x567)
        x569=self.sigmoid36(x568)
        x570=operator.mul(x569, x564)
        return x570

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x567 = torch.randn(torch.Size([batch_size, 76, 1, 1]))
x564 = torch.randn(torch.Size([batch_size, 1824, 7, 7]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x567, x564)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
