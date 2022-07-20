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
        self.batchnorm2d22 = BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d39 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x114, x101):
        x115=self.batchnorm2d22(x114)
        x116=operator.add(x115, x101)
        x117=self.conv2d39(x116)
        x118=self.batchnorm2d23(x117)
        return x118

m = M().eval()
x114 = torch.randn(torch.Size([1, 40, 28, 28]))
x101 = torch.randn(torch.Size([1, 40, 28, 28]))
start = time.time()
output = m(x114, x101)
end = time.time()
print(end-start)
