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
        self.conv2d40 = Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d32 = BatchNorm2d(112, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d46 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x119, x135):
        x120=self.conv2d40(x119)
        x121=self.batchnorm2d32(x120)
        x136=operator.add(x135, x121)
        x137=self.conv2d46(x136)
        return x137

m = M().eval()
x119 = torch.randn(torch.Size([1, 480, 14, 14]))
x135 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x119, x135)
end = time.time()
print(end-start)
