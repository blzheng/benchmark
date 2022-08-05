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

    def forward(self, x352, x354, x358, x356, x357):
        x359=x352.view(x354, 2, x358, x356, x357)
        x360=torch.transpose(x359, 1, 2)
        return x360

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x352 = torch.randn(torch.Size([batch_size, 464, 7, 7]))
x354 = batch_size
x358 = 232
x356 = 7
x357 = 7
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x352, x354, x358, x356, x357)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
