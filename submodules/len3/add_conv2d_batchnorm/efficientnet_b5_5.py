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
        self.conv2d38 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d22 = BatchNorm2d(240, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x117, x102):
        x118=operator.add(x117, x102)
        x119=self.conv2d38(x118)
        x120=self.batchnorm2d22(x119)
        return x120

m = M().eval()
x117 = torch.randn(torch.Size([1, 40, 56, 56]))
x102 = torch.randn(torch.Size([1, 40, 56, 56]))
start = time.time()
output = m(x117, x102)
end = time.time()
print(end-start)
