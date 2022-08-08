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
        self.conv2d90 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid18 = Sigmoid()
        self.conv2d91 = Conv2d(480, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d53 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d92 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x282, x279):
        x283=self.conv2d90(x282)
        x284=self.sigmoid18(x283)
        x285=operator.mul(x284, x279)
        x286=self.conv2d91(x285)
        x287=self.batchnorm2d53(x286)
        x288=self.conv2d92(x287)
        return x288

m = M().eval()
x282 = torch.randn(torch.Size([1, 20, 1, 1]))
x279 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x282, x279)
end = time.time()
print(end-start)
