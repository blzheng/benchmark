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
        self.layernorm21 = LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        self.linear42 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu21 = GELU(approximate='none')

    def forward(self, x252):
        x253=self.layernorm21(x252)
        x254=self.linear42(x253)
        x255=self.gelu21(x254)
        return x255

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x252 = torch.randn(torch.Size([batch_size, 14, 14, 384]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x252)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
