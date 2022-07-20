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
        self.batchnorm2d9 = BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d18 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d10 = BatchNorm2d(240, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x53):
        x54=self.batchnorm2d9(x53)
        x55=self.conv2d18(x54)
        x56=self.batchnorm2d10(x55)
        return x56

m = M().eval()
x53 = torch.randn(torch.Size([1, 40, 56, 56]))
start = time.time()
output = m(x53)
end = time.time()
print(end-start)
