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

    def forward(self, x104, x109):
        x110=operator.mul(x104, x109)
        x111=self.conv2d36(x110)
        return x111

m = M().eval()
x104 = torch.randn(torch.Size([1, 288, 56, 56]))
x109 = torch.randn(torch.Size([1, 288, 1, 1]))
start = time.time()
output = m(x104, x109)
end = time.time()
print(end-start)
