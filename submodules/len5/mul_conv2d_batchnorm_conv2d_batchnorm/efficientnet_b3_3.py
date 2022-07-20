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
        self.conv2d43 = Conv2d(288, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d25 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d44 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x130, x125):
        x131=operator.mul(x130, x125)
        x132=self.conv2d43(x131)
        x133=self.batchnorm2d25(x132)
        x134=self.conv2d44(x133)
        x135=self.batchnorm2d26(x134)
        return x135

m = M().eval()
x130 = torch.randn(torch.Size([1, 288, 1, 1]))
x125 = torch.randn(torch.Size([1, 288, 14, 14]))
start = time.time()
output = m(x130, x125)
end = time.time()
print(end-start)
