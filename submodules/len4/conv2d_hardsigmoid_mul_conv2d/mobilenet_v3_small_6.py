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
        self.conv2d39 = Conv2d(72, 288, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid6 = Hardsigmoid()
        self.conv2d40 = Conv2d(288, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x113, x110):
        x114=self.conv2d39(x113)
        x115=self.hardsigmoid6(x114)
        x116=operator.mul(x115, x110)
        x117=self.conv2d40(x116)
        return x117

m = M().eval()
x113 = torch.randn(torch.Size([1, 72, 1, 1]))
x110 = torch.randn(torch.Size([1, 288, 7, 7]))
start = time.time()
output = m(x113, x110)
end = time.time()
print(end-start)
