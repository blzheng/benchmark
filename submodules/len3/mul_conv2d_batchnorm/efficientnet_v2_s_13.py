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
        self.conv2d88 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d60 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x280, x275):
        x281=operator.mul(x280, x275)
        x282=self.conv2d88(x281)
        x283=self.batchnorm2d60(x282)
        return x283

m = M().eval()
x280 = torch.randn(torch.Size([1, 960, 1, 1]))
x275 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x280, x275)
end = time.time()
print(end-start)
