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
        self.conv2d45 = Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d35 = BatchNorm2d(112, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x132, x127, x121):
        x133=operator.mul(x132, x127)
        x134=self.conv2d45(x133)
        x135=self.batchnorm2d35(x134)
        x136=operator.add(x135, x121)
        return x136

m = M().eval()
x132 = torch.randn(torch.Size([1, 672, 1, 1]))
x127 = torch.randn(torch.Size([1, 672, 14, 14]))
x121 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x132, x127, x121)
end = time.time()
print(end-start)
