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
        self.batchnorm2d35 = BatchNorm2d(72, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.conv2d36 = Conv2d(72, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(432, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x101, x94):
        x102=self.batchnorm2d35(x101)
        x103=operator.add(x102, x94)
        x104=self.conv2d36(x103)
        x105=self.batchnorm2d36(x104)
        return x105

m = M().eval()
x101 = torch.randn(torch.Size([1, 72, 14, 14]))
x94 = torch.randn(torch.Size([1, 72, 14, 14]))
start = time.time()
output = m(x101, x94)
end = time.time()
print(end-start)
