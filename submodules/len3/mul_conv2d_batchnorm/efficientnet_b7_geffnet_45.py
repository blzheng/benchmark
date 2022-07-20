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
        self.conv2d226 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d134 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x670, x675):
        x676=operator.mul(x670, x675)
        x677=self.conv2d226(x676)
        x678=self.batchnorm2d134(x677)
        return x678

m = M().eval()
x670 = torch.randn(torch.Size([1, 2304, 7, 7]))
x675 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x670, x675)
end = time.time()
print(end-start)
