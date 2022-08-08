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
        self.conv2d78 = Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d79 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x229, x225, x219):
        x230=x229.sigmoid()
        x231=operator.mul(x225, x230)
        x232=self.conv2d78(x231)
        x233=self.batchnorm2d46(x232)
        x234=operator.add(x233, x219)
        x235=self.conv2d79(x234)
        x236=self.batchnorm2d47(x235)
        return x236

m = M().eval()
x229 = torch.randn(torch.Size([1, 672, 1, 1]))
x225 = torch.randn(torch.Size([1, 672, 14, 14]))
x219 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x229, x225, x219)
end = time.time()
print(end-start)
