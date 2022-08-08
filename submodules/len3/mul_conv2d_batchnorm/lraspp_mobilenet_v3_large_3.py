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
        self.conv2d40 = Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d32 = BatchNorm2d(112, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x116, x111):
        x117=operator.mul(x116, x111)
        x118=self.conv2d40(x117)
        x119=self.batchnorm2d32(x118)
        return x119

m = M().eval()
x116 = torch.randn(torch.Size([1, 480, 1, 1]))
x111 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x116, x111)
end = time.time()
print(end-start)
