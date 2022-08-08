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
        self.conv2d2 = Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d2 = BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d3 = Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d3 = BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x8, x5):
        x9=self.conv2d2(x8)
        x10=self.batchnorm2d2(x9)
        x11=operator.add(x10, x5)
        x12=self.conv2d3(x11)
        x13=self.batchnorm2d3(x12)
        return x13

m = M().eval()
x8 = torch.randn(torch.Size([1, 16, 112, 112]))
x5 = torch.randn(torch.Size([1, 16, 112, 112]))
start = time.time()
output = m(x8, x5)
end = time.time()
print(end-start)
