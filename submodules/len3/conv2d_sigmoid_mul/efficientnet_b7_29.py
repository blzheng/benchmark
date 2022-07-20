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
        self.conv2d145 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid29 = Sigmoid()

    def forward(self, x454, x451):
        x455=self.conv2d145(x454)
        x456=self.sigmoid29(x455)
        x457=operator.mul(x456, x451)
        return x457

m = M().eval()
x454 = torch.randn(torch.Size([1, 56, 1, 1]))
x451 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x454, x451)
end = time.time()
print(end-start)
