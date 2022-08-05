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
        self.layernorm5 = LayerNorm((192,), eps=1e-06, elementwise_affine=True)
        self.linear10 = Linear(in_features=192, out_features=768, bias=True)
        self.gelu5 = GELU(approximate=none)

    def forward(self, x70):
        x71=self.layernorm5(x70)
        x72=self.linear10(x71)
        x73=self.gelu5(x72)
        return x73

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x70 = torch.randn(torch.Size([batch_size, 28, 28, 192]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x70)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
