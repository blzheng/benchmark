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
        self.hardsigmoid2 = Hardsigmoid()
        self.conv2d20 = Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x55, x51):
        x56=self.hardsigmoid2(x55)
        x57=operator.mul(x56, x51)
        x58=self.conv2d20(x57)
        return x58

m = M().eval()
x55 = torch.randn(torch.Size([1, 240, 1, 1]))
x51 = torch.randn(torch.Size([1, 240, 14, 14]))
start = time.time()
output = m(x55, x51)
end = time.time()
print(end-start)
