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
        self.conv2d46 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x134, x139):
        x140=operator.mul(x134, x139)
        x141=self.conv2d46(x140)
        x142=self.batchnorm2d26(x141)
        return x142

m = M().eval()
x134 = torch.randn(torch.Size([1, 288, 56, 56]))
x139 = torch.randn(torch.Size([1, 288, 1, 1]))
start = time.time()
output = m(x134, x139)
end = time.time()
print(end-start)
