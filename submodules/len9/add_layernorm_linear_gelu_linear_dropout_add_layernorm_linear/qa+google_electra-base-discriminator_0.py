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
        self.layernorm1 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear4 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear5 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout3 = Dropout(p=0.1, inplace=False)
        self.layernorm2 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear6 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x62, x28):
        x63=operator.add(x62, x28)
        x64=self.layernorm1(x63)
        x65=self.linear4(x64)
        x66=torch._C._nn.gelu(x65)
        x67=self.linear5(x66)
        x68=self.dropout3(x67)
        x69=operator.add(x68, x64)
        x70=self.layernorm2(x69)
        x71=self.linear6(x70)
        return x71

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x62 = torch.randn(torch.Size([batch_size, 384, 768]))
x28 = torch.randn(torch.Size([batch_size, 384, 768]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x62, x28)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
