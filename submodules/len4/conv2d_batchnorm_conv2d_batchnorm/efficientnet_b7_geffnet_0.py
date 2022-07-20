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
        self.conv2d21 = Conv2d(192, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d22 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x66):
        x67=self.conv2d21(x66)
        x68=self.batchnorm2d11(x67)
        x69=self.conv2d22(x68)
        x70=self.batchnorm2d12(x69)
        return x70

m = M().eval()
x66 = torch.randn(torch.Size([1, 192, 56, 56]))
start = time.time()
output = m(x66)
end = time.time()
print(end-start)
