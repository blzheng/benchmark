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
        self.conv2d74 = Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d44 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x211, x216, x205):
        x217=operator.mul(x211, x216)
        x218=self.conv2d74(x217)
        x219=self.batchnorm2d44(x218)
        x220=operator.add(x219, x205)
        return x220

m = M().eval()
x211 = torch.randn(torch.Size([1, 1152, 7, 7]))
x216 = torch.randn(torch.Size([1, 1152, 1, 1]))
x205 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x211, x216, x205)
end = time.time()
print(end-start)
