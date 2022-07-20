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
        self.conv2d180 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid36 = Sigmoid()
        self.conv2d181 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x566, x563):
        x567=self.conv2d180(x566)
        x568=self.sigmoid36(x567)
        x569=operator.mul(x568, x563)
        x570=self.conv2d181(x569)
        return x570

m = M().eval()
x566 = torch.randn(torch.Size([1, 56, 1, 1]))
x563 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x566, x563)
end = time.time()
print(end-start)
