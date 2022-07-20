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
        self.sigmoid25 = Sigmoid()
        self.conv2d134 = Conv2d(888, 888, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d82 = BatchNorm2d(888, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x421, x417, x411):
        x422=self.sigmoid25(x421)
        x423=operator.mul(x422, x417)
        x424=self.conv2d134(x423)
        x425=self.batchnorm2d82(x424)
        x426=operator.add(x411, x425)
        return x426

m = M().eval()
x421 = torch.randn(torch.Size([1, 888, 1, 1]))
x417 = torch.randn(torch.Size([1, 888, 7, 7]))
x411 = torch.randn(torch.Size([1, 888, 7, 7]))
start = time.time()
output = m(x421, x417, x411)
end = time.time()
print(end-start)
