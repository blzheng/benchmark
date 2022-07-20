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
        self.batchnorm2d26 = BatchNorm2d(96, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d41 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(576, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x117):
        x118=self.batchnorm2d26(x117)
        x119=self.conv2d41(x118)
        x120=self.batchnorm2d27(x119)
        return x120

m = M().eval()
x117 = torch.randn(torch.Size([1, 96, 7, 7]))
start = time.time()
output = m(x117)
end = time.time()
print(end-start)
