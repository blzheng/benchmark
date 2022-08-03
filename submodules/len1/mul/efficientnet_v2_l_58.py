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

    def forward(self, x1048, x1043):
        x1049=operator.mul(x1048, x1043)
        return x1049

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x1048 = torch.randn(torch.Size([batch_size, 3840, 1, 1]))
x1043 = torch.randn(torch.Size([batch_size, 3840, 7, 7]))
start_time=time.time()
for i in range(10):
    output = m(x1048, x1043)
total_iter_time = time.time() - start_time
Throughput = batch_size * 10 / total_iter_time
file_current = os.path.basename(__file__)
print(file_current,',',BS,',',Throughput) 
