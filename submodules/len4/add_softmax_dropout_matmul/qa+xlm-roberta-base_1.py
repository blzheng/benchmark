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
        self.dropout4 = Dropout(p=0.1, inplace=False)
        self._tensor_constant20 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)

    def forward(self, x91, x83):
        x93=operator.add(x91, self._tensor_constant20)
        x94=torch.nn.functional.softmax(x93,dim=-1, _stacklevel=3, dtype=None)
        x95=self.dropout4(x94)
        x96=torch.matmul(x95, x83)
        return x96

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x91 = torch.randn(torch.Size([batch_size, 12, 384, 384]))
x83 = torch.randn(torch.Size([batch_size, 12, 384, 64]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x91, x83)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
