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
        self.conv2d35 = Conv2d(144, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(48, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d36 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x100, x95, x89):
        x101=operator.mul(x100, x95)
        x102=self.conv2d35(x101)
        x103=self.batchnorm2d23(x102)
        x104=operator.add(x103, x89)
        x105=self.conv2d36(x104)
        return x105

m = M().eval()
x100 = torch.randn(torch.Size([1, 144, 1, 1]))
x95 = torch.randn(torch.Size([1, 144, 14, 14]))
x89 = torch.randn(torch.Size([1, 48, 14, 14]))
start = time.time()
output = m(x100, x95, x89)
end = time.time()
print(end-start)
