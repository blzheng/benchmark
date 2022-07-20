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
        self.relu142 = ReLU(inplace=True)
        self.conv2d142 = Conv2d(1776, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x504):
        x505=self.relu142(x504)
        x506=self.conv2d142(x505)
        return x506

m = M().eval()
x504 = torch.randn(torch.Size([1, 1776, 7, 7]))
start = time.time()
output = m(x504)
end = time.time()
print(end-start)
