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
        self.conv2d12 = Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d13 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x39, x32):
        x40=self.conv2d12(x39)
        x41=self.batchnorm2d12(x40)
        x42=operator.add(x41, x32)
        x43=self.conv2d13(x42)
        return x43

m = M().eval()
x39 = torch.randn(torch.Size([1, 24, 56, 56]))
x32 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x39, x32)
end = time.time()
print(end-start)
