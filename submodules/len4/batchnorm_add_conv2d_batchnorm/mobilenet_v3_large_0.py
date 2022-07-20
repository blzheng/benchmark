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
        self.batchnorm2d2 = BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d3 = Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d3 = BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x7, x3):
        x8=self.batchnorm2d2(x7)
        x9=operator.add(x8, x3)
        x10=self.conv2d3(x9)
        x11=self.batchnorm2d3(x10)
        return x11

m = M().eval()
x7 = torch.randn(torch.Size([1, 16, 112, 112]))
x3 = torch.randn(torch.Size([1, 16, 112, 112]))
start = time.time()
output = m(x7, x3)
end = time.time()
print(end-start)
