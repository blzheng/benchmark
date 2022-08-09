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
        self.linear64 = Linear(in_features=256, out_features=256, bias=True)
        self.dropout32 = Dropout(p=0.1, inplace=False)
        self.layernorm21 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear65 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear66 = Linear(in_features=1024, out_features=256, bias=True)

    def forward(self, x477, x480, x449):
        x481=x477.view(x480)
        x482=self.linear64(x481)
        x483=self.dropout32(x482)
        x484=operator.add(x483, x449)
        x485=self.layernorm21(x484)
        x486=self.linear65(x485)
        x487=torch._C._nn.gelu(x486)
        x488=self.linear66(x487)
        return x488

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x477 = torch.randn(torch.Size([batch_size, 384, 4, 64]))
x480 = (batch_size, 384, 256, )
x449 = torch.randn(torch.Size([batch_size, 384, 256]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x477, x480, x449)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
