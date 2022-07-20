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
        self.conv2d57 = Conv2d(432, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x175, x170):
        x176=operator.mul(x175, x170)
        x177=self.conv2d57(x176)
        return x177

m = M().eval()
x175 = torch.randn(torch.Size([1, 432, 1, 1]))
x170 = torch.randn(torch.Size([1, 432, 28, 28]))
start = time.time()
output = m(x175, x170)
end = time.time()
print(end-start)
