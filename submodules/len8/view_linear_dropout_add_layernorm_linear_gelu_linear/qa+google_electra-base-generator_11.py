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
        self.linear70 = Linear(in_features=256, out_features=256, bias=True)
        self.dropout35 = Dropout(p=0.1, inplace=False)
        self.layernorm23 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear71 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear72 = Linear(in_features=1024, out_features=256, bias=True)

    def forward(self, x519, x522, x491):
        x523=x519.view(x522)
        x524=self.linear70(x523)
        x525=self.dropout35(x524)
        x526=operator.add(x525, x491)
        x527=self.layernorm23(x526)
        x528=self.linear71(x527)
        x529=torch._C._nn.gelu(x528)
        x530=self.linear72(x529)
        return x530

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x519 = torch.randn(torch.Size([batch_size, 384, 4, 64]))
x522 = (batch_size, 384, 256, )
x491 = torch.randn(torch.Size([batch_size, 384, 256]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x519, x522, x491)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
