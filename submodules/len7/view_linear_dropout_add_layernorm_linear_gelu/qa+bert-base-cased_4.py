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
        self.linear27 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout14 = Dropout(p=0.1, inplace=False)
        self.layernorm9 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear28 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x224, x227, x196):
        x228=x224.view(x227)
        x229=self.linear27(x228)
        x230=self.dropout14(x229)
        x231=operator.add(x230, x196)
        x232=self.layernorm9(x231)
        x233=self.linear28(x232)
        x234=torch._C._nn.gelu(x233)
        return x234

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x224 = torch.randn(torch.Size([batch_size, 384, 12, 64]))
x227 = (batch_size, 384, 768, )
x196 = torch.randn(torch.Size([batch_size, 384, 768]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x224, x227, x196)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
