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
        self.conv2d87 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.avgpool2d2 = AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x311):
        x312=self.conv2d87(x311)
        x313=self.avgpool2d2(x312)
        return x313

m = M().eval()
x311 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x311)
end = time.time()
print(end-start)
