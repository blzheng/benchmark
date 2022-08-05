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

    def forward(self, x308, x310, x314, x312, x313):
        x315=x308.view(x310, 2, x314, x312, x313)
        x316=torch.transpose(x315, 1, 2)
        return x316

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x308 = torch.randn(torch.Size([batch_size, 704, 7, 7]))
x310 = batch_size
x314 = 352
x312 = 7
x313 = 7
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x308, x310, x314, x312, x313)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
