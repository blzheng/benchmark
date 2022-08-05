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
        self.conv2d4 = Conv2d(224, 8, kernel_size=(1, 1), stride=(1, 1))
        self.relu3 = ReLU()
        self.conv2d5 = Conv2d(8, 224, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid0 = Sigmoid()

    def forward(self, x12):
        x13=self.conv2d4(x12)
        x14=self.relu3(x13)
        x15=self.conv2d5(x14)
        x16=self.sigmoid0(x15)
        return x16

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x12 = torch.randn(torch.Size([batch_size, 224, 1, 1]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x12)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
