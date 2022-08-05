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
        self.dropout0 = Dropout(p=0.5, inplace=False)
        self.conv2d25 = Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x58, x60):
        x61=torch.cat([x58, x60], 1)
        x62=self.dropout0(x61)
        x63=self.conv2d25(x62)
        return x63

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x58 = torch.randn(torch.Size([batch_size, 256, 13, 13]))
x60 = torch.randn(torch.Size([batch_size, 256, 13, 13]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x58, x60)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
