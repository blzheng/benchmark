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
        self.conv2d45 = Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d35 = BatchNorm2d(112, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d46 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x130, x125, x119):
        x131=operator.mul(x130, x125)
        x132=self.conv2d45(x131)
        x133=self.batchnorm2d35(x132)
        x134=operator.add(x133, x119)
        x135=self.conv2d46(x134)
        return x135

m = M().eval()
x130 = torch.randn(torch.Size([1, 672, 1, 1]))
x125 = torch.randn(torch.Size([1, 672, 14, 14]))
x119 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x130, x125, x119)
end = time.time()
print(end-start)
