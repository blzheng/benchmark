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
        self.linear65 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear66 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout33 = Dropout(p=0.1, inplace=False)

    def forward(self, x485):
        x486=self.linear65(x485)
        x487=torch._C._nn.gelu(x486)
        x488=self.linear66(x487)
        x489=self.dropout33(x488)
        return x489

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x485 = torch.randn(torch.Size([batch_size, 384, 256]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x485)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
