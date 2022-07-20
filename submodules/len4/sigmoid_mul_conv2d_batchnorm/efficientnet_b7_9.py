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
        self.sigmoid9 = Sigmoid()
        self.conv2d46 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(48, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x141, x137):
        x142=self.sigmoid9(x141)
        x143=operator.mul(x142, x137)
        x144=self.conv2d46(x143)
        x145=self.batchnorm2d26(x144)
        return x145

m = M().eval()
x141 = torch.randn(torch.Size([1, 288, 1, 1]))
x137 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x141, x137)
end = time.time()
print(end-start)
