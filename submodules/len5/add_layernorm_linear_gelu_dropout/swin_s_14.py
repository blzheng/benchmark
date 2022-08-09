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
        self.layernorm32 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear30 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu14 = GELU(approximate='none')
        self.dropout28 = Dropout(p=0.0, inplace=False)

    def forward(self, x355, x369):
        x370=operator.add(x355, x369)
        x371=self.layernorm32(x370)
        x372=self.linear30(x371)
        x373=self.gelu14(x372)
        x374=self.dropout28(x373)
        return x374

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x355 = torch.randn(torch.Size([batch_size, 14, 14, 384]))
x369 = torch.randn(torch.Size([49, 49, batch_size2]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x355, x369)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
