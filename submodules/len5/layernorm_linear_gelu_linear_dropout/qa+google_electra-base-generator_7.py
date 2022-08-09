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
        self.layernorm15 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear47 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear48 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout24 = Dropout(p=0.1, inplace=False)

    def forward(self, x358):
        x359=self.layernorm15(x358)
        x360=self.linear47(x359)
        x361=torch._C._nn.gelu(x360)
        x362=self.linear48(x361)
        x363=self.dropout24(x362)
        return x363

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x358 = torch.randn(torch.Size([batch_size, 384, 256]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x358)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
