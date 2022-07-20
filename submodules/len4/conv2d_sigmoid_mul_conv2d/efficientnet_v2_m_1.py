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
        self.conv2d32 = Conv2d(40, 640, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid1 = Sigmoid()
        self.conv2d33 = Conv2d(640, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x107, x104):
        x108=self.conv2d32(x107)
        x109=self.sigmoid1(x108)
        x110=operator.mul(x109, x104)
        x111=self.conv2d33(x110)
        return x111

m = M().eval()
x107 = torch.randn(torch.Size([1, 40, 1, 1]))
x104 = torch.randn(torch.Size([1, 640, 14, 14]))
start = time.time()
output = m(x107, x104)
end = time.time()
print(end-start)