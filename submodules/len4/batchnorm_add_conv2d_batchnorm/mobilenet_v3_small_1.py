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
        self.batchnorm2d8 = BatchNorm2d(24, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d11 = Conv2d(24, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(96, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x29, x22):
        x30=self.batchnorm2d8(x29)
        x31=operator.add(x30, x22)
        x32=self.conv2d11(x31)
        x33=self.batchnorm2d9(x32)
        return x33

m = M().eval()
x29 = torch.randn(torch.Size([1, 24, 28, 28]))
x22 = torch.randn(torch.Size([1, 24, 28, 28]))
start = time.time()
output = m(x29, x22)
end = time.time()
print(end-start)
