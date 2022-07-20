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
        self.conv2d11 = Conv2d(16, 144, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid1 = Sigmoid()
        self.conv2d12 = Conv2d(144, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x32, x29):
        x33=self.conv2d11(x32)
        x34=self.sigmoid1(x33)
        x35=operator.mul(x34, x29)
        x36=self.conv2d12(x35)
        return x36

m = M().eval()
x32 = torch.randn(torch.Size([1, 16, 1, 1]))
x29 = torch.randn(torch.Size([1, 144, 28, 28]))
start = time.time()
output = m(x32, x29)
end = time.time()
print(end-start)
