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
        self.batchnorm2d1 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu61 = ReLU6(inplace=True)
        self.conv2d2 = Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d2 = BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d3 = Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d3 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x4):
        x5=self.batchnorm2d1(x4)
        x6=self.relu61(x5)
        x7=self.conv2d2(x6)
        x8=self.batchnorm2d2(x7)
        x9=self.conv2d3(x8)
        x10=self.batchnorm2d3(x9)
        return x10

m = M().eval()
x4 = torch.randn(torch.Size([1, 32, 112, 112]))
start = time.time()
output = m(x4)
end = time.time()
print(end-start)
