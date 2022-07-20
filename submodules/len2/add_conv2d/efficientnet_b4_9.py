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
        self.conv2d69 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x212, x197):
        x213=operator.add(x212, x197)
        x214=self.conv2d69(x213)
        return x214

m = M().eval()
x212 = torch.randn(torch.Size([1, 112, 14, 14]))
x197 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x212, x197)
end = time.time()
print(end-start)
