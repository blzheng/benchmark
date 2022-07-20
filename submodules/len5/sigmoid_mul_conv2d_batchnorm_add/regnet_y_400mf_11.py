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
        self.sigmoid11 = Sigmoid()
        self.conv2d64 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x197, x193, x187):
        x198=self.sigmoid11(x197)
        x199=operator.mul(x198, x193)
        x200=self.conv2d64(x199)
        x201=self.batchnorm2d40(x200)
        x202=operator.add(x187, x201)
        return x202

m = M().eval()
x197 = torch.randn(torch.Size([1, 440, 1, 1]))
x193 = torch.randn(torch.Size([1, 440, 7, 7]))
x187 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x197, x193, x187)
end = time.time()
print(end-start)
