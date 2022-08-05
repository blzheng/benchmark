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
        self.relu27 = ReLU()
        self.conv2d36 = Conv2d(54, 216, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid6 = Sigmoid()

    def forward(self, x111, x109):
        x112=self.relu27(x111)
        x113=self.conv2d36(x112)
        x114=self.sigmoid6(x113)
        x115=operator.mul(x114, x109)
        return x115

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x111 = torch.randn(torch.Size([batch_size, 54, 1, 1]))
x109 = torch.randn(torch.Size([batch_size, 216, 28, 28]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x111, x109)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
