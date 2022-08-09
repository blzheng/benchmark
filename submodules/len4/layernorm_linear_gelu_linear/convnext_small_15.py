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
        self.layernorm15 = LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        self.linear30 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu15 = GELU(approximate='none')
        self.linear31 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x186):
        x187=self.layernorm15(x186)
        x188=self.linear30(x187)
        x189=self.gelu15(x188)
        x190=self.linear31(x189)
        return x190

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x186 = torch.randn(torch.Size([batch_size, 14, 14, 384]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x186)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
