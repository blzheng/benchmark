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
        self.conv2d7 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d7 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = ReLU(inplace=True)
        self.conv2d8 = Conv2d(336, 672, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d8 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x21, x15):
        x22=self.conv2d7(x21)
        x23=self.batchnorm2d7(x22)
        x24=operator.add(x15, x23)
        x25=self.relu6(x24)
        x26=self.conv2d8(x25)
        x27=self.batchnorm2d8(x26)
        return x27

m = M().eval()
x21 = torch.randn(torch.Size([1, 336, 56, 56]))
x15 = torch.randn(torch.Size([1, 336, 56, 56]))
start = time.time()
output = m(x21, x15)
end = time.time()
print(end-start)
