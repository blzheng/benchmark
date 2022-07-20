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
        self.conv2d9 = Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d10 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x21, x26):
        x27=operator.mul(x21, x26)
        x28=self.conv2d9(x27)
        x29=self.batchnorm2d5(x28)
        x30=self.conv2d10(x29)
        return x30

m = M().eval()
x21 = torch.randn(torch.Size([1, 96, 56, 56]))
x26 = torch.randn(torch.Size([1, 96, 1, 1]))
start = time.time()
output = m(x21, x26)
end = time.time()
print(end-start)
