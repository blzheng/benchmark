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
        self.maxpool2d3 = MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.conv2d14 = Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x27, x33, x39, x43):
        x44=torch.cat([x27, x33, x39, x43], 1)
        x60=self.maxpool2d3(x44)
        x61=self.conv2d14(x60)
        x62=self.batchnorm2d14(x61)
        return x62

m = M().eval()
x27 = torch.randn(torch.Size([1, 64, 28, 28]))
x33 = torch.randn(torch.Size([1, 128, 28, 28]))
x39 = torch.randn(torch.Size([1, 32, 28, 28]))
x43 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x27, x33, x39, x43)
end = time.time()
print(end-start)
