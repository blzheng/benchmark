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
        self.relu76 = ReLU(inplace=True)
        self.conv2d99 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x297, x311):
        x312=operator.add(x297, x311)
        x313=self.relu76(x312)
        x314=self.conv2d99(x313)
        return x314

m = M().eval()
x297 = torch.randn(torch.Size([1, 336, 14, 14]))
x311 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x297, x311)
end = time.time()
print(end-start)
