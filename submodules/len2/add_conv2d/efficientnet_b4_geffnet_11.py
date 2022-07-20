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
        self.conv2d79 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x234, x220):
        x235=operator.add(x234, x220)
        x236=self.conv2d79(x235)
        return x236

m = M().eval()
x234 = torch.randn(torch.Size([1, 112, 14, 14]))
x220 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x234, x220)
end = time.time()
print(end-start)