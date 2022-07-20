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
        self.conv2d8 = Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x22, x16):
        x23=self.conv2d8(x22)
        x24=self.batchnorm2d8(x23)
        x25=operator.add(x16, x24)
        return x25

m = M().eval()
x22 = torch.randn(torch.Size([1, 144, 56, 56]))
x16 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x22, x16)
end = time.time()
print(end-start)
