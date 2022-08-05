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
        self.relu21 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(216, 216, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=9, bias=False)

    def forward(self, x89):
        x90=self.relu21(x89)
        x91=self.conv2d29(x90)
        return x91

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x89 = torch.randn(torch.Size([batch_size, 216, 28, 28]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x89)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
