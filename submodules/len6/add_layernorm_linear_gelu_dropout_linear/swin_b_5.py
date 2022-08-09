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
        self.layernorm14 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear12 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu5 = GELU(approximate='none')
        self.dropout10 = Dropout(p=0.0, inplace=False)
        self.linear13 = Linear(in_features=2048, out_features=512, bias=True)

    def forward(self, x148, x162):
        x163=operator.add(x148, x162)
        x164=self.layernorm14(x163)
        x165=self.linear12(x164)
        x166=self.gelu5(x165)
        x167=self.dropout10(x166)
        x168=self.linear13(x167)
        return x168

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x148 = torch.randn(torch.Size([batch_size, 14, 14, 512]))
x162 = torch.randn(torch.Size([49, 49, batch_size6]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x148, x162)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
