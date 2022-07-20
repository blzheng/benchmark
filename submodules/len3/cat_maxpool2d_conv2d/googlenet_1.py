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
        self.maxpool2d6 = MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.conv2d26 = Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x68, x74, x80, x84):
        x85=torch.cat([x68, x74, x80, x84], 1)
        x101=self.maxpool2d6(x85)
        x102=self.conv2d26(x101)
        return x102

m = M().eval()
x68 = torch.randn(torch.Size([1, 192, 14, 14]))
x74 = torch.randn(torch.Size([1, 208, 14, 14]))
x80 = torch.randn(torch.Size([1, 48, 14, 14]))
x84 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x68, x74, x80, x84)
end = time.time()
print(end-start)
