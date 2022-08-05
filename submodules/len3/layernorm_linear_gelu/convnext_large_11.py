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
        self.layernorm11 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.linear22 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu11 = GELU(approximate=none)

    def forward(self, x142):
        x143=self.layernorm11(x142)
        x144=self.linear22(x143)
        x145=self.gelu11(x144)
        return x145

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x142 = torch.randn(torch.Size([batch_size, 14, 14, 768]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x142)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
