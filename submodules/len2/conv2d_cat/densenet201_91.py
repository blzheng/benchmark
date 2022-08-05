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
        self.conv2d193 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x683, x481, x488, x495, x502, x509, x516, x523, x530, x537, x544, x551, x558, x565, x572, x579, x586, x593, x600, x607, x614, x621, x628, x635, x642, x649, x656, x663, x670, x677, x691):
        x684=self.conv2d193(x683)
        x692=torch.cat([x481, x488, x495, x502, x509, x516, x523, x530, x537, x544, x551, x558, x565, x572, x579, x586, x593, x600, x607, x614, x621, x628, x635, x642, x649, x656, x663, x670, x677, x684, x691], 1)
        return x692

m = M().eval()

CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x683 = torch.randn(torch.Size([batch_size, 128, 7, 7]))
x481 = torch.randn(torch.Size([batch_size, 896, 7, 7]))
x488 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x495 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x502 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x509 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x516 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x523 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x530 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x537 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x544 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x551 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x558 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x565 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x572 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x579 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x586 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x593 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x600 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x607 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x614 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x621 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x628 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x635 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x642 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x649 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x656 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x663 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x670 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x677 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
x691 = torch.randn(torch.Size([batch_size, 32, 7, 7]))
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x683, x481, x488, x495, x502, x509, x516, x523, x530, x537, x544, x551, x558, x565, x572, x579, x586, x593, x600, x607, x614, x621, x628, x635, x642, x649, x656, x663, x670, x677, x691)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
