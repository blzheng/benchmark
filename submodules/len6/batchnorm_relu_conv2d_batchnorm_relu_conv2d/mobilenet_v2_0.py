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
        self.batchnorm2d0 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu60 = ReLU6(inplace=True)
        self.conv2d1 = Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d1 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu61 = ReLU6(inplace=True)
        self.conv2d2 = Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x1):
        x2=self.batchnorm2d0(x1)
        x3=self.relu60(x2)
        x4=self.conv2d1(x3)
        x5=self.batchnorm2d1(x4)
        x6=self.relu61(x5)
        x7=self.conv2d2(x6)
        return x7

m = M().eval()
x1 = torch.randn(torch.Size([1, 32, 112, 112]))
start = time.time()
output = m(x1)
end = time.time()
print(end-start)
