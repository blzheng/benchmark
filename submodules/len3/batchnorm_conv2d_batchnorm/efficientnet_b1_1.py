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
        self.batchnorm2d7 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d14 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x40):
        x41=self.batchnorm2d7(x40)
        x42=self.conv2d14(x41)
        x43=self.batchnorm2d8(x42)
        return x43

m = M().eval()
x40 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x40)
end = time.time()
print(end-start)
