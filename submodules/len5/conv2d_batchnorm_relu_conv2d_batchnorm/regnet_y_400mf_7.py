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
        self.conv2d34 = Conv2d(208, 208, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d22 = BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)
        self.conv2d35 = Conv2d(208, 208, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=26, bias=False)
        self.batchnorm2d23 = BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x105):
        x106=self.conv2d34(x105)
        x107=self.batchnorm2d22(x106)
        x108=self.relu25(x107)
        x109=self.conv2d35(x108)
        x110=self.batchnorm2d23(x109)
        return x110

m = M().eval()
x105 = torch.randn(torch.Size([1, 208, 14, 14]))
start = time.time()
output = m(x105)
end = time.time()
print(end-start)
