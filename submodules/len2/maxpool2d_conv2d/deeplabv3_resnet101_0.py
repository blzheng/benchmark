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
        self.maxpool2d0 = MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.conv2d1 = Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x5):
        x6=self.maxpool2d0(x5)
        x7=self.conv2d1(x6)
        return x7

m = M().eval()
x5 = torch.randn(torch.Size([1, 64, 112, 112]))
start = time.time()
output = m(x5)
end = time.time()
print(end-start)
