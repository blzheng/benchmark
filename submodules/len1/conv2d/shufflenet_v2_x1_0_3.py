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
        self.conv2d3 = Conv2d(24, 58, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x4):
        x10=self.conv2d3(x4)
        return x10

m = M().eval()
x4 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x4)
end = time.time()
print(end-start)
