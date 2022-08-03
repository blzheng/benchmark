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
        self.sigmoid15 = Sigmoid()
        self.conv2d79 = Conv2d(1152, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d80 = Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x238, x234):
        x239=self.sigmoid15(x238)
        x240=operator.mul(x239, x234)
        x241=self.conv2d79(x240)
        x242=self.batchnorm2d47(x241)
        x243=self.conv2d80(x242)
        x244=self.batchnorm2d48(x243)
        return x244

m = M().eval()
x238 = torch.randn(torch.Size([1, 1152, 1, 1]))
x234 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x238, x234)
end = time.time()
print(end-start)
