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
        self.conv2d3 = Conv2d(24, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x12, x8):
        x13=operator.add(x12, x8)
        x14=self.conv2d3(x13)
        return x14

m = M().eval()
x12 = torch.randn(torch.Size([1, 24, 112, 112]))
x8 = torch.randn(torch.Size([1, 24, 112, 112]))
start = time.time()
output = m(x12, x8)
end = time.time()
print(end-start)
