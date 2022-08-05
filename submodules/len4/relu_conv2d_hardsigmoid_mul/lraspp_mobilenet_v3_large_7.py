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

    def forward(self, x171, x169):
        x172=self.relu18(x171)
        x173=self.conv2d59(x172)
        x174=self.hardsigmoid7(x173)
        x175=operator.mul(x174, x169)
        return x175

m = M().eval()
x171 = torch.randn(torch.Size([1, 240, 1, 1]))
x169 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x171, x169)
end = time.time()
print(end-start)
