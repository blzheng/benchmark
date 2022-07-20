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
        self.conv2d8 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x31, x4, x11, x18, x25, x39):
        x32=self.conv2d8(x31)
        x40=torch.cat([x4, x11, x18, x25, x32, x39], 1)
        x41=self.batchnorm2d11(x40)
        return x41

m = M().eval()
x31 = torch.randn(torch.Size([1, 128, 56, 56]))
x4 = torch.randn(torch.Size([1, 64, 56, 56]))
x11 = torch.randn(torch.Size([1, 32, 56, 56]))
x18 = torch.randn(torch.Size([1, 32, 56, 56]))
x25 = torch.randn(torch.Size([1, 32, 56, 56]))
x39 = torch.randn(torch.Size([1, 32, 56, 56]))
start = time.time()
output = m(x31, x4, x11, x18, x25, x39)
end = time.time()
print(end-start)
