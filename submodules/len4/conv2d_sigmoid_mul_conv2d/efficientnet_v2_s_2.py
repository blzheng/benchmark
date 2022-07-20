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
        self.conv2d32 = Conv2d(32, 512, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid2 = Sigmoid()
        self.conv2d33 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x104, x101):
        x105=self.conv2d32(x104)
        x106=self.sigmoid2(x105)
        x107=operator.mul(x106, x101)
        x108=self.conv2d33(x107)
        return x108

m = M().eval()
x104 = torch.randn(torch.Size([1, 32, 1, 1]))
x101 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x104, x101)
end = time.time()
print(end-start)
