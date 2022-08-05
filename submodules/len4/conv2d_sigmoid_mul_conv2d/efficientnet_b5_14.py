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
        self.conv2d71 = Conv2d(32, 768, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid14 = Sigmoid()
        self.conv2d72 = Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x219, x216):
        x220=self.conv2d71(x219)
        x221=self.sigmoid14(x220)
        x222=operator.mul(x221, x216)
        x223=self.conv2d72(x222)
        return x223

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x219 = torch.randn(torch.Size([batch_size, 32, 1, 1]))
x216 = torch.randn(torch.Size([batch_size, 768, 14, 14]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x219, x216)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
