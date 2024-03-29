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
        self.conv2d98 = Conv2d(1392, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d60 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu76 = ReLU(inplace=True)
        self.conv2d99 = Conv2d(1392, 3712, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d61 = BatchNorm2d(3712, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x309, x297):
        x310=self.conv2d98(x309)
        x311=self.batchnorm2d60(x310)
        x312=operator.add(x297, x311)
        x313=self.relu76(x312)
        x314=self.conv2d99(x313)
        x315=self.batchnorm2d61(x314)
        return x315

m = M().eval()
x309 = torch.randn(torch.Size([1, 1392, 14, 14]))
x297 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x309, x297)
end = time.time()
print(end-start)
