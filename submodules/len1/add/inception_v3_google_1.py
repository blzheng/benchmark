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

    def forward(self, x7):
        x8=operator.add(x7, -0.08799999999999997)
        return x8

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x7 = torch.randn(torch.Size([batch_size, 1, 224, 224]))
start_time=time.time()
for i in range(10):
    output = m(x7)
total_iter_time = time.time() - start_time
Throughput = batch_size * 10 / total_iter_time
file_current = os.path.basename(__file__)
print(file_current,',',BS,',',Throughput) 
