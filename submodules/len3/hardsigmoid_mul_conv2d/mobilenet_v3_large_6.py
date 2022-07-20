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
        self.hardsigmoid6 = Hardsigmoid()
        self.conv2d55 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x158, x154):
        x159=self.hardsigmoid6(x158)
        x160=operator.mul(x159, x154)
        x161=self.conv2d55(x160)
        return x161

m = M().eval()
x158 = torch.randn(torch.Size([1, 960, 1, 1]))
x154 = torch.randn(torch.Size([1, 960, 7, 7]))
start = time.time()
output = m(x158, x154)
end = time.time()
print(end-start)
