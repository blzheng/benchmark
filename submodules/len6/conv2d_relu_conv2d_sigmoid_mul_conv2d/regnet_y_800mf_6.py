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
        self.conv2d36 = Conv2d(320, 80, kernel_size=(1, 1), stride=(1, 1))
        self.relu27 = ReLU()
        self.conv2d37 = Conv2d(80, 320, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid6 = Sigmoid()
        self.conv2d38 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x112, x111):
        x113=self.conv2d36(x112)
        x114=self.relu27(x113)
        x115=self.conv2d37(x114)
        x116=self.sigmoid6(x115)
        x117=operator.mul(x116, x111)
        x118=self.conv2d38(x117)
        return x118

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x112 = torch.randn(torch.Size([batch_size, 320, 1, 1]))
x111 = torch.randn(torch.Size([batch_size, 320, 14, 14]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x112, x111)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
