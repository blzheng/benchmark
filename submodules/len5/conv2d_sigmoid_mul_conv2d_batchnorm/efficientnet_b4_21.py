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
        self.conv2d107 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid21 = Sigmoid()
        self.conv2d108 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d64 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x332, x329):
        x333=self.conv2d107(x332)
        x334=self.sigmoid21(x333)
        x335=operator.mul(x334, x329)
        x336=self.conv2d108(x335)
        x337=self.batchnorm2d64(x336)
        return x337

m = M().eval()
x332 = torch.randn(torch.Size([1, 40, 1, 1]))
x329 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x332, x329)
end = time.time()
print(end-start)
