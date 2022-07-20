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
        self.sigmoid12 = Sigmoid()
        self.conv2d63 = Conv2d(528, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d64 = Conv2d(120, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x191, x187):
        x192=self.sigmoid12(x191)
        x193=operator.mul(x192, x187)
        x194=self.conv2d63(x193)
        x195=self.batchnorm2d37(x194)
        x196=self.conv2d64(x195)
        return x196

m = M().eval()
x191 = torch.randn(torch.Size([1, 528, 1, 1]))
x187 = torch.randn(torch.Size([1, 528, 14, 14]))
start = time.time()
output = m(x191, x187)
end = time.time()
print(end-start)
