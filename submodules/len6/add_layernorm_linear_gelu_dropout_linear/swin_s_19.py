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
        self.layernorm42 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear40 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu19 = GELU(approximate='none')
        self.dropout38 = Dropout(p=0.0, inplace=False)
        self.linear41 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x470, x484):
        x485=operator.add(x470, x484)
        x486=self.layernorm42(x485)
        x487=self.linear40(x486)
        x488=self.gelu19(x487)
        x489=self.dropout38(x488)
        x490=self.linear41(x489)
        return x490

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x470 = torch.randn(torch.Size([batch_size, 14, 14, 384]))
x484 = torch.randn(torch.Size([49, 49, batch_size2]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x470, x484)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
