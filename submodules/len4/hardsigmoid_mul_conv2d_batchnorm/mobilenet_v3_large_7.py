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
        self.hardsigmoid7 = Hardsigmoid()
        self.conv2d60 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d44 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x173, x169):
        x174=self.hardsigmoid7(x173)
        x175=operator.mul(x174, x169)
        x176=self.conv2d60(x175)
        x177=self.batchnorm2d44(x176)
        return x177

m = M().eval()
x173 = torch.randn(torch.Size([1, 960, 1, 1]))
x169 = torch.randn(torch.Size([1, 960, 7, 7]))
start = time.time()
output = m(x173, x169)
end = time.time()
print(end-start)
