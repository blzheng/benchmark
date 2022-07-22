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
        self.conv2d3 = Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d3 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = ReLU(inplace=True)
        self.conv2d5 = Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x10, x14):
        x11=self.conv2d3(x10)
        x12=self.batchnorm2d3(x11)
        x15=operator.add(x12, x14)
        x16=self.relu1(x15)
        x17=self.conv2d5(x16)
        x18=self.batchnorm2d5(x17)
        return x18

m = M().eval()
x10 = torch.randn(torch.Size([1, 128, 56, 56]))
x14 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x10, x14)
end = time.time()
print(end-start)
