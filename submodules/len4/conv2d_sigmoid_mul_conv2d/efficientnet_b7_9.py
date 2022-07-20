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
        self.conv2d45 = Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid9 = Sigmoid()
        self.conv2d46 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x140, x137):
        x141=self.conv2d45(x140)
        x142=self.sigmoid9(x141)
        x143=operator.mul(x142, x137)
        x144=self.conv2d46(x143)
        return x144

m = M().eval()
x140 = torch.randn(torch.Size([1, 12, 1, 1]))
x137 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x140, x137)
end = time.time()
print(end-start)
