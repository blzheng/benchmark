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
        self.conv2d8 = Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(24, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d9 = Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(72, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x25, x19):
        x26=self.conv2d8(x25)
        x27=self.batchnorm2d8(x26)
        x28=operator.add(x27, x19)
        x29=self.conv2d9(x28)
        x30=self.batchnorm2d9(x29)
        return x30

m = M().eval()
x25 = torch.randn(torch.Size([1, 72, 56, 56]))
x19 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x25, x19)
end = time.time()
print(end-start)
