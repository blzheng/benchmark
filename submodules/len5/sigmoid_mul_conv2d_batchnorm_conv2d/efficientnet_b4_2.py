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
        self.sigmoid6 = Sigmoid()
        self.conv2d33 = Conv2d(192, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d19 = BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d34 = Conv2d(56, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x99, x95):
        x100=self.sigmoid6(x99)
        x101=operator.mul(x100, x95)
        x102=self.conv2d33(x101)
        x103=self.batchnorm2d19(x102)
        x104=self.conv2d34(x103)
        return x104

m = M().eval()
x99 = torch.randn(torch.Size([1, 192, 1, 1]))
x95 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x99, x95)
end = time.time()
print(end-start)
