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
        self.conv2d145 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x515, x369, x376, x383, x390, x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467, x474, x481, x488, x495, x502, x509, x523):
        x516=self.conv2d145(x515)
        x524=torch.cat([x369, x376, x383, x390, x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467, x474, x481, x488, x495, x502, x509, x516, x523], 1)
        return x524

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x515 = torch.randn(torch.Size([batch_size, 128, 7, 7]))
x369 = torch.randn(torch.Size([batch_size, 640, 7, 7]))
x376 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x383 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x390 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x397 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x404 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x411 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x418 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x425 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x432 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x439 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x446 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x453 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x460 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x467 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x474 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x481 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x488 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x495 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x502 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x509 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x523 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x515, x369, x376, x383, x390, x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467, x474, x481, x488, x495, x502, x509, x523)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
