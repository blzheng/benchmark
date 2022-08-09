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
        self.linear10 = Linear(in_features=256, out_features=256, bias=True)
        self.dropout5 = Dropout(p=0.1, inplace=False)
        self.layernorm3 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear11 = Linear(in_features=256, out_features=1024, bias=True)

    def forward(self, x99, x102, x71):
        x103=x99.view(x102)
        x104=self.linear10(x103)
        x105=self.dropout5(x104)
        x106=operator.add(x105, x71)
        x107=self.layernorm3(x106)
        x108=self.linear11(x107)
        return x108

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x99 = torch.randn(torch.Size([batch_size, 384, 4, 64]))
x102 = (batch_size, 384, 256, )
x71 = torch.randn(torch.Size([batch_size, 384, 256]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x99, x102, x71)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
