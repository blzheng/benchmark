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
        self.relu24 = ReLU(inplace=True)
        self.conv2d33 = Conv2d(696, 696, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x87, x101):
        x102=operator.add(x87, x101)
        x103=self.relu24(x102)
        x104=self.conv2d33(x103)
        return x104

m = M().eval()
x87 = torch.randn(torch.Size([1, 696, 28, 28]))
x101 = torch.randn(torch.Size([1, 696, 28, 28]))
start = time.time()
output = m(x87, x101)
end = time.time()
print(end-start)
