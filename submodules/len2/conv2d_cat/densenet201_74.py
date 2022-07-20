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

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv2d159 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x564, x481, x488, x495, x502, x509, x516, x523, x530, x537, x544, x551, x558, x572):
        x565=self.conv2d159(x564)
        x573=torch.cat([x481, x488, x495, x502, x509, x516, x523, x530, x537, x544, x551, x558, x565, x572], 1)
        return x573

m = M().eval()
x564 = torch.randn(torch.Size([1, 128, 7, 7]))
x481 = torch.randn(torch.Size([1, 896, 7, 7]))
x488 = torch.randn(torch.Size([1, 32, 7, 7]))
x495 = torch.randn(torch.Size([1, 32, 7, 7]))
x502 = torch.randn(torch.Size([1, 32, 7, 7]))
x509 = torch.randn(torch.Size([1, 32, 7, 7]))
x516 = torch.randn(torch.Size([1, 32, 7, 7]))
x523 = torch.randn(torch.Size([1, 32, 7, 7]))
x530 = torch.randn(torch.Size([1, 32, 7, 7]))
x537 = torch.randn(torch.Size([1, 32, 7, 7]))
x544 = torch.randn(torch.Size([1, 32, 7, 7]))
x551 = torch.randn(torch.Size([1, 32, 7, 7]))
x558 = torch.randn(torch.Size([1, 32, 7, 7]))
x572 = torch.randn(torch.Size([1, 32, 7, 7]))
start = time.time()
output = m(x564, x481, x488, x495, x502, x509, x516, x523, x530, x537, x544, x551, x558, x572)
end = time.time()
print(end-start)
