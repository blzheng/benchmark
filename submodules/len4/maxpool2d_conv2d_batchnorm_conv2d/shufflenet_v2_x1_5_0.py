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
        self.maxpool2d0 = MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.conv2d1 = Conv2d(24, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=24, bias=False)
        self.batchnorm2d1 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d2 = Conv2d(24, 88, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x3):
        x4=self.maxpool2d0(x3)
        x5=self.conv2d1(x4)
        x6=self.batchnorm2d1(x5)
        x7=self.conv2d2(x6)
        return x7

m = M().eval()
x3 = torch.randn(torch.Size([1, 24, 112, 112]))
start = time.time()
output = m(x3)
end = time.time()
print(end-start)
