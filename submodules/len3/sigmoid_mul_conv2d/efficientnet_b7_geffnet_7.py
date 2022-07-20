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
        self.conv2d36 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x108, x104):
        x109=x108.sigmoid()
        x110=operator.mul(x104, x109)
        x111=self.conv2d36(x110)
        return x111

m = M().eval()
x108 = torch.randn(torch.Size([1, 288, 1, 1]))
x104 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x108, x104)
end = time.time()
print(end-start)
