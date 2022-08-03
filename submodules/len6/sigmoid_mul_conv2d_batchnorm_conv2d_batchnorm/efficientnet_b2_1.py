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
        self.sigmoid2 = Sigmoid()
        self.conv2d13 = Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d7 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d14 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x37, x33):
        x38=self.sigmoid2(x37)
        x39=operator.mul(x38, x33)
        x40=self.conv2d13(x39)
        x41=self.batchnorm2d7(x40)
        x42=self.conv2d14(x41)
        x43=self.batchnorm2d8(x42)
        return x43

m = M().eval()
x37 = torch.randn(torch.Size([1, 96, 1, 1]))
x33 = torch.randn(torch.Size([1, 96, 56, 56]))
start = time.time()
output = m(x37, x33)
end = time.time()
print(end-start)
