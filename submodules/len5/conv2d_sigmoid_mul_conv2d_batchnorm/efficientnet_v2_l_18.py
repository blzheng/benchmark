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
        self.conv2d126 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid18 = Sigmoid()
        self.conv2d127 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d89 = BatchNorm2d(224, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x410, x407):
        x411=self.conv2d126(x410)
        x412=self.sigmoid18(x411)
        x413=operator.mul(x412, x407)
        x414=self.conv2d127(x413)
        x415=self.batchnorm2d89(x414)
        return x415

m = M().eval()
x410 = torch.randn(torch.Size([1, 56, 1, 1]))
x407 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x410, x407)
end = time.time()
print(end-start)
