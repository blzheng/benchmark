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
        self.conv2d8 = Conv2d(336, 672, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d8 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu9 = ReLU(inplace=True)
        self.conv2d12 = Conv2d(672, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x25, x35):
        x26=self.conv2d8(x25)
        x27=self.batchnorm2d8(x26)
        x36=operator.add(x27, x35)
        x37=self.relu9(x36)
        x38=self.conv2d12(x37)
        x39=self.batchnorm2d12(x38)
        return x39

m = M().eval()
x25 = torch.randn(torch.Size([1, 336, 56, 56]))
x35 = torch.randn(torch.Size([1, 672, 28, 28]))
start = time.time()
output = m(x25, x35)
end = time.time()
print(end-start)