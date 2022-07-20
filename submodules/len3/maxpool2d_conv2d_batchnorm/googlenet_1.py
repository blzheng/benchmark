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
        self.maxpool2d1 = MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv2d3 = Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d3 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x23):
        x24=self.maxpool2d1(x23)
        x25=self.conv2d3(x24)
        x26=self.batchnorm2d3(x25)
        return x26

m = M().eval()
x23 = torch.randn(torch.Size([1, 192, 56, 56]))
start = time.time()
output = m(x23)
end = time.time()
print(end-start)
