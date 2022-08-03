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
        self.conv2d3 = Conv2d(10, 40, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid0 = Sigmoid()
        self.conv2d4 = Conv2d(40, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d2 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x9, x6, x26):
        x10=self.conv2d3(x9)
        x11=self.sigmoid0(x10)
        x12=operator.mul(x11, x6)
        x13=self.conv2d4(x12)
        x14=self.batchnorm2d2(x13)
        x27=operator.add(x26, x14)
        return x27

m = M().eval()
x9 = torch.randn(torch.Size([1, 10, 1, 1]))
x6 = torch.randn(torch.Size([1, 40, 112, 112]))
x26 = torch.randn(torch.Size([1, 24, 112, 112]))
start = time.time()
output = m(x9, x6, x26)
end = time.time()
print(end-start)
