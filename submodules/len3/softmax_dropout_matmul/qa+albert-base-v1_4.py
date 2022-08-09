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
        self.dropout1 = Dropout(p=0.1, inplace=False)

    def forward(self, x200, x195):
        x201=torch.nn.functional.softmax(x200,dim=-1, _stacklevel=3, dtype=None)
        x202=self.dropout1(x201)
        x203=torch.matmul(x202, x195)
        return x203

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x200 = torch.randn(torch.Size([batch_size, 12, 384, 384]))
x195 = torch.randn(torch.Size([batch_size, 12, 384, 64]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x200, x195)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
