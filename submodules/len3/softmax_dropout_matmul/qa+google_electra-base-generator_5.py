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
        self.dropout16 = Dropout(p=0.1, inplace=False)

    def forward(self, x262, x252):
        x263=torch.nn.functional.softmax(x262,dim=-1, _stacklevel=3, dtype=None)
        x264=self.dropout16(x263)
        x265=torch.matmul(x264, x252)
        return x265

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x262 = torch.randn(torch.Size([batch_size, 4, 384, 384]))
x252 = torch.randn(torch.Size([batch_size, 4, 384, 64]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x262, x252)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
