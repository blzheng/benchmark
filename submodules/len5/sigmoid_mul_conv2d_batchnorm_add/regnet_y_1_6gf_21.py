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
        self.sigmoid21 = Sigmoid()
        self.conv2d113 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d69 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x355, x351, x345):
        x356=self.sigmoid21(x355)
        x357=operator.mul(x356, x351)
        x358=self.conv2d113(x357)
        x359=self.batchnorm2d69(x358)
        x360=operator.add(x345, x359)
        return x360

m = M().eval()
x355 = torch.randn(torch.Size([1, 336, 1, 1]))
x351 = torch.randn(torch.Size([1, 336, 14, 14]))
x345 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x355, x351, x345)
end = time.time()
print(end-start)
