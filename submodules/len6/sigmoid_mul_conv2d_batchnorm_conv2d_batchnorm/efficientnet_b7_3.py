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
        self.sigmoid18 = Sigmoid()
        self.conv2d91 = Conv2d(480, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d53 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d92 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d54 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x283, x279):
        x284=self.sigmoid18(x283)
        x285=operator.mul(x284, x279)
        x286=self.conv2d91(x285)
        x287=self.batchnorm2d53(x286)
        x288=self.conv2d92(x287)
        x289=self.batchnorm2d54(x288)
        return x289

m = M().eval()
x283 = torch.randn(torch.Size([1, 480, 1, 1]))
x279 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x283, x279)
end = time.time()
print(end-start)
