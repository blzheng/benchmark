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
        self.conv2d46 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(576, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x132, x118):
        x133=operator.add(x132, x118)
        x134=self.conv2d46(x133)
        x135=self.batchnorm2d30(x134)
        return x135

m = M().eval()
x132 = torch.randn(torch.Size([1, 96, 7, 7]))
x118 = torch.randn(torch.Size([1, 96, 7, 7]))
start = time.time()
output = m(x132, x118)
end = time.time()
print(end-start)
