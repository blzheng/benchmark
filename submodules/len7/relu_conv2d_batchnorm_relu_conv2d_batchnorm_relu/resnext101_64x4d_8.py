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
        self.relu25 = ReLU(inplace=True)
        self.conv2d31 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)
        self.conv2d32 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        self.batchnorm2d32 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x99):
        x100=self.relu25(x99)
        x101=self.conv2d31(x100)
        x102=self.batchnorm2d31(x101)
        x103=self.relu28(x102)
        x104=self.conv2d32(x103)
        x105=self.batchnorm2d32(x104)
        x106=self.relu28(x105)
        return x106

m = M().eval()
x99 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x99)
end = time.time()
print(end-start)
