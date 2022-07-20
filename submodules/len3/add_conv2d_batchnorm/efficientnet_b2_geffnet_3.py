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
        self.conv2d34 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x100, x86):
        x101=operator.add(x100, x86)
        x102=self.conv2d34(x101)
        x103=self.batchnorm2d20(x102)
        return x103

m = M().eval()
x100 = torch.randn(torch.Size([1, 48, 28, 28]))
x86 = torch.randn(torch.Size([1, 48, 28, 28]))
start = time.time()
output = m(x100, x86)
end = time.time()
print(end-start)
