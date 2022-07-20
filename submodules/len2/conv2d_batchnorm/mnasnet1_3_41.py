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
        self.conv2d41 = Conv2d(1488, 248, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(248, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x117):
        x118=self.conv2d41(x117)
        x119=self.batchnorm2d41(x118)
        return x119

m = M().eval()
x117 = torch.randn(torch.Size([1, 1488, 7, 7]))
start = time.time()
output = m(x117)
end = time.time()
print(end-start)
