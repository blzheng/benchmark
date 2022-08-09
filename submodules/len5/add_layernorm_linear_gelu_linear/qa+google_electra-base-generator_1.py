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
        self.layernorm3 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear11 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear12 = Linear(in_features=1024, out_features=256, bias=True)

    def forward(self, x105, x71):
        x106=operator.add(x105, x71)
        x107=self.layernorm3(x106)
        x108=self.linear11(x107)
        x109=torch._C._nn.gelu(x108)
        x110=self.linear12(x109)
        return x110

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x105 = torch.randn(torch.Size([batch_size, 384, 256]))
x71 = torch.randn(torch.Size([batch_size, 384, 256]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x105, x71)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
