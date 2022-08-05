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

    def forward(self, x130, x132, x136, x134, x135):
        x137=x130.view(x132, 2, x136, x134, x135)
        x138=torch.transpose(x137, 1, 2)
        return x138

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x130 = torch.randn(torch.Size([batch_size, 352, 14, 14]))
x132 = batch_size
x136 = batch_size76
x134 = batch_size4
x135 = batch_size4
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x130, x132, x136, x134, x135)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
