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
        self.conv2d16 = Conv2d(384, 48, kernel_size=(1, 1), stride=(1, 1))
        self.relu16 = ReLU(inplace=True)
        self.conv2d17 = Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x39):
        x40=self.conv2d16(x39)
        x41=self.relu16(x40)
        x42=self.conv2d17(x41)
        return x42

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x39 = torch.randn(torch.Size([batch_size, 384, 27, 27]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x39)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
