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
        self.conv2d103 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x368, x313, x320, x327, x334, x341, x348, x355, x362, x376):
        x369=self.conv2d103(x368)
        x377=torch.cat([x313, x320, x327, x334, x341, x348, x355, x362, x369, x376], 1)
        return x377

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x368 = torch.randn(torch.Size([batch_size, 128, 7, 7]))
x313 = torch.randn(torch.Size([batch_size, 512, 7, 7]))
x320 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x327 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x334 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x341 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x348 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x355 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x362 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x376 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x368, x313, x320, x327, x334, x341, x348, x355, x362, x376)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
