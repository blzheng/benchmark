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
        self.gelu15 = GELU(approximate='none')
        self.dropout30 = Dropout(p=0.0, inplace=False)
        self.linear33 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x395):
        x396=self.gelu15(x395)
        x397=self.dropout30(x396)
        x398=self.linear33(x397)
        return x398

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x395 = torch.randn(torch.Size([batch_size, 12, 49, 49]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x395)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
