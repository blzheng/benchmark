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
        self.layernorm20 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)

    def forward(self, x217, x231):
        x232=operator.add(x217, x231)
        x233=self.layernorm20(x232)
        return x233

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x217 = torch.randn(torch.Size([batch_size, 14, 14, 384]))
x231 = torch.randn(torch.Size([49, 49, batch_size2]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x217, x231)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
