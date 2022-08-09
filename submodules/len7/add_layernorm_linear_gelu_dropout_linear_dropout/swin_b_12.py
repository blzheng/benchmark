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
        self.layernorm28 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear26 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu12 = GELU(approximate='none')
        self.dropout24 = Dropout(p=0.0, inplace=False)
        self.linear27 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout25 = Dropout(p=0.0, inplace=False)

    def forward(self, x309, x323):
        x324=operator.add(x309, x323)
        x325=self.layernorm28(x324)
        x326=self.linear26(x325)
        x327=self.gelu12(x326)
        x328=self.dropout24(x327)
        x329=self.linear27(x328)
        x330=self.dropout25(x329)
        return x330

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x309 = torch.randn(torch.Size([batch_size, 14, 14, 512]))
x323 = torch.randn(torch.Size([49, 49, batch_size6]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x309, x323)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
