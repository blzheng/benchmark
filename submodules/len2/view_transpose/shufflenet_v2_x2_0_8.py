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

    def forward(self, x196, x198, x202, x200, x201):
        x203=x196.view(x198, 2, x202, x200, x201)
        x204=torch.transpose(x203, 1, 2)
        return x204

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x196 = torch.randn(torch.Size([batch_size, 488, 14, 14]))
x198 = batch_size
x202 = 244
x200 = batch_size4
x201 = batch_size4
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x196, x198, x202, x200, x201)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
