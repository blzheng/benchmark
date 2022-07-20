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
        self.conv2d110 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid22 = Sigmoid()
        self.conv2d111 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d65 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x344, x341):
        x345=self.conv2d110(x344)
        x346=self.sigmoid22(x345)
        x347=operator.mul(x346, x341)
        x348=self.conv2d111(x347)
        x349=self.batchnorm2d65(x348)
        return x349

m = M().eval()
x344 = torch.randn(torch.Size([1, 40, 1, 1]))
x341 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x344, x341)
end = time.time()
print(end-start)
