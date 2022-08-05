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
        self.relu7 = ReLU(inplace=True)
        self.conv2d8 = Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        self.relu8 = ReLU(inplace=True)

    def forward(self, x19):
        x20=self.relu7(x19)
        x21=self.conv2d8(x20)
        x22=self.relu8(x21)
        return x22

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x19 = torch.randn(torch.Size([batch_size, 32, 27, 27]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x19)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
