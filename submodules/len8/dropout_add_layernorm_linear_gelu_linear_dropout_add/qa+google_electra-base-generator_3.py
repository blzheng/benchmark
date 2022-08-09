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
        self.dropout11 = Dropout(p=0.1, inplace=False)
        self.layernorm7 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear23 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear24 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout12 = Dropout(p=0.1, inplace=False)

    def forward(self, x188, x155):
        x189=self.dropout11(x188)
        x190=operator.add(x189, x155)
        x191=self.layernorm7(x190)
        x192=self.linear23(x191)
        x193=torch._C._nn.gelu(x192)
        x194=self.linear24(x193)
        x195=self.dropout12(x194)
        x196=operator.add(x195, x191)
        return x196

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x188 = torch.randn(torch.Size([batch_size, 384, 256]))
x155 = torch.randn(torch.Size([batch_size, 384, 256]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x188, x155)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
