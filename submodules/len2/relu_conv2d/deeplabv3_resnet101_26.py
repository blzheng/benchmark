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
        self.relu25 = ReLU(inplace=True)
        self.conv2d31 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x101):
        x102=self.relu25(x101)
        x103=self.conv2d31(x102)
        return x103

m = M().eval()
x101 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x101)
end = time.time()
print(end-start)
