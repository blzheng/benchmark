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
        self.conv2d56 = Conv2d(336, 84, kernel_size=(1, 1), stride=(1, 1))
        self.relu43 = ReLU()
        self.conv2d57 = Conv2d(84, 336, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid10 = Sigmoid()

    def forward(self, x176):
        x177=self.conv2d56(x176)
        x178=self.relu43(x177)
        x179=self.conv2d57(x178)
        x180=self.sigmoid10(x179)
        return x180

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x176 = torch.randn(torch.Size([batch_size, 336, 1, 1]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x176)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
