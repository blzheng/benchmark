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
        self.conv2d77 = Conv2d(432, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(144, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d78 = Conv2d(144, 864, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(864, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x239, x234):
        x240=operator.mul(x239, x234)
        x241=self.conv2d77(x240)
        x242=self.batchnorm2d45(x241)
        x243=self.conv2d78(x242)
        x244=self.batchnorm2d46(x243)
        return x244

m = M().eval()
x239 = torch.randn(torch.Size([1, 432, 1, 1]))
x234 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x239, x234)
end = time.time()
print(end-start)
