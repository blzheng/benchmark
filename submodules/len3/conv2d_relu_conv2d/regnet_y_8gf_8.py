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
        self.conv2d46 = Conv2d(896, 224, kernel_size=(1, 1), stride=(1, 1))
        self.relu35 = ReLU()
        self.conv2d47 = Conv2d(224, 896, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x144):
        x145=self.conv2d46(x144)
        x146=self.relu35(x145)
        x147=self.conv2d47(x146)
        return x147

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x144 = torch.randn(torch.Size([batch_size, 896, 1, 1]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x144)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
