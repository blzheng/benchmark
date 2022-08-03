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
        self.relu11 = ReLU()
        self.conv2d39 = Conv2d(72, 288, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid6 = Hardsigmoid()
        self.conv2d40 = Conv2d(288, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(96, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d41 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x112, x110):
        x113=self.relu11(x112)
        x114=self.conv2d39(x113)
        x115=self.hardsigmoid6(x114)
        x116=operator.mul(x115, x110)
        x117=self.conv2d40(x116)
        x118=self.batchnorm2d26(x117)
        x119=self.conv2d41(x118)
        return x119

m = M().eval()
x112 = torch.randn(torch.Size([1, 72, 1, 1]))
x110 = torch.randn(torch.Size([1, 288, 7, 7]))
start = time.time()
output = m(x112, x110)
end = time.time()
print(end-start)
