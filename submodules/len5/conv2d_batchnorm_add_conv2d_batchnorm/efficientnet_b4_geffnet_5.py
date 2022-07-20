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
        self.conv2d153 = Conv2d(1632, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d91 = BatchNorm2d(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d159 = Conv2d(448, 1792, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d95 = BatchNorm2d(1792, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x455, x471):
        x456=self.conv2d153(x455)
        x457=self.batchnorm2d91(x456)
        x472=operator.add(x471, x457)
        x473=self.conv2d159(x472)
        x474=self.batchnorm2d95(x473)
        return x474

m = M().eval()
x455 = torch.randn(torch.Size([1, 1632, 7, 7]))
x471 = torch.randn(torch.Size([1, 448, 7, 7]))
start = time.time()
output = m(x455, x471)
end = time.time()
print(end-start)
