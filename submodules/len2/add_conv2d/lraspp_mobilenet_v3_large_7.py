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
        self.conv2d46 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x133, x119):
        x134=operator.add(x133, x119)
        x135=self.conv2d46(x134)
        return x135

m = M().eval()
x133 = torch.randn(torch.Size([1, 112, 14, 14]))
x119 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x133, x119)
end = time.time()
print(end-start)
