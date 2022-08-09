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
        self.layernorm33 = LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
        self.linear66 = Linear(in_features=1024, out_features=4096, bias=True)
        self.gelu33 = GELU(approximate='none')
        self.linear67 = Linear(in_features=4096, out_features=1024, bias=True)

    def forward(self, x390):
        x391=self.layernorm33(x390)
        x392=self.linear66(x391)
        x393=self.gelu33(x392)
        x394=self.linear67(x393)
        return x394

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x390 = torch.randn(torch.Size([batch_size, 7, 7, 1024]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x390)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
