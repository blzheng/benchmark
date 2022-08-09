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
        self.dropout18 = Dropout(p=0.1, inplace=False)
        self.layernorm12 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear36 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x277, x274):
        x278=self.dropout18(x277)
        x279=operator.add(x278, x274)
        x280=self.layernorm12(x279)
        x281=self.linear36(x280)
        return x281

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x277 = torch.randn(torch.Size([batch_size, 384, 768]))
x274 = torch.randn(torch.Size([batch_size, 384, 768]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x277, x274)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
