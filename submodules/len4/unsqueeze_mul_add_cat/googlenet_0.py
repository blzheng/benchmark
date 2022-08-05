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

    def forward(self, x1, x8, x12):
        x2=torch.unsqueeze(x1, 1)
        x3=operator.mul(x2, 0.458)
        x4=operator.add(x3, -0.030000000000000027)
        x13=torch.cat((x4, x8, x12), 1)
        return x13

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x1 = torch.randn(torch.Size([batch_size, 224, 224]))
x8 = torch.randn(torch.Size([batch_size, 1, 224, 224]))
x12 = torch.randn(torch.Size([batch_size, 1, 224, 224]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x1, x8, x12)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
