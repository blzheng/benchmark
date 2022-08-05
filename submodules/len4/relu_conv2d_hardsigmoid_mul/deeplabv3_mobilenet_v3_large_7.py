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
        self.relu18 = ReLU()
        self.conv2d59 = Conv2d(240, 960, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid7 = Hardsigmoid()

    def forward(self, x173, x171):
        x174=self.relu18(x173)
        x175=self.conv2d59(x174)
        x176=self.hardsigmoid7(x175)
        x177=operator.mul(x176, x171)
        return x177

m = M().eval()
x173 = torch.randn(torch.Size([1, 240, 1, 1]))
x171 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x173, x171)
end = time.time()
print(end-start)
