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
        self.conv2d108 = Conv2d(1152, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d64 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d109 = Conv2d(320, 1920, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d65 = BatchNorm2d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x332, x327):
        x333=operator.mul(x332, x327)
        x334=self.conv2d108(x333)
        x335=self.batchnorm2d64(x334)
        x336=self.conv2d109(x335)
        x337=self.batchnorm2d65(x336)
        return x337

m = M().eval()
x332 = torch.randn(torch.Size([1, 1152, 1, 1]))
x327 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x332, x327)
end = time.time()
print(end-start)
