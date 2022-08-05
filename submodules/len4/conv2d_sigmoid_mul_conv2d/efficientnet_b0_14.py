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
        self.conv2d73 = Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid14 = Sigmoid()
        self.conv2d74 = Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x221, x218):
        x222=self.conv2d73(x221)
        x223=self.sigmoid14(x222)
        x224=operator.mul(x223, x218)
        x225=self.conv2d74(x224)
        return x225

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x221 = torch.randn(torch.Size([batch_size, 48, 1, 1]))
x218 = torch.randn(torch.Size([batch_size, 1152, 7, 7]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x221, x218)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
