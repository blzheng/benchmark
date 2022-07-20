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
        self.batchnorm2d59 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d60 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d60 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x193, x185):
        x194=self.batchnorm2d59(x193)
        x195=operator.add(x194, x185)
        x196=self.conv2d60(x195)
        x197=self.batchnorm2d60(x196)
        return x197

m = M().eval()
x193 = torch.randn(torch.Size([1, 192, 7, 7]))
x185 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x193, x185)
end = time.time()
print(end-start)
