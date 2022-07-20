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
        self.conv2d28 = Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d16 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x78, x83, x72):
        x84=operator.mul(x78, x83)
        x85=self.conv2d28(x84)
        x86=self.batchnorm2d16(x85)
        x87=operator.add(x86, x72)
        return x87

m = M().eval()
x78 = torch.randn(torch.Size([1, 192, 56, 56]))
x83 = torch.randn(torch.Size([1, 192, 1, 1]))
x72 = torch.randn(torch.Size([1, 32, 56, 56]))
start = time.time()
output = m(x78, x83, x72)
end = time.time()
print(end-start)
