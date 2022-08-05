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
        self.conv2d237 = Conv2d(128, 3072, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid42 = Sigmoid()
        self.conv2d238 = Conv2d(3072, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x757, x754):
        x758=self.conv2d237(x757)
        x759=self.sigmoid42(x758)
        x760=operator.mul(x759, x754)
        x761=self.conv2d238(x760)
        return x761

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x757 = torch.randn(torch.Size([batch_size, 128, 1, 1]))
x754 = torch.randn(torch.Size([batch_size, 3072, 7, 7]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x757, x754)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
