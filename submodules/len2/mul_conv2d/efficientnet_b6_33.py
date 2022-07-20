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
        self.conv2d167 = Conv2d(2064, 344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x521, x516):
        x522=operator.mul(x521, x516)
        x523=self.conv2d167(x522)
        return x523

m = M().eval()
x521 = torch.randn(torch.Size([1, 2064, 1, 1]))
x516 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x521, x516)
end = time.time()
print(end-start)
