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
        self.layernorm10 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.linear20 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu10 = GELU(approximate=none)
        self.linear21 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x131):
        x132=self.layernorm10(x131)
        x133=self.linear20(x132)
        x134=self.gelu10(x133)
        x135=self.linear21(x134)
        return x135

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x131 = torch.randn(torch.Size([batch_size, 14, 14, 768]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x131)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
