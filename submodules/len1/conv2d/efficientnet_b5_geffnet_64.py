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
        self.conv2d20 = Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x195):
        x196=self.conv2d20(x195)
        return x196

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x195 = torch.randn(torch.Size([batch_size, 384, 14, 14]))
start_time=time.time()
for i in range(10):
    output = m(x195)
total_iter_time = time.time() - start_time
Throughput = batch_size * 10 / total_iter_time
file_current = os.path.basename(__file__)
print(file_current,',',BS,',',Throughput) 
