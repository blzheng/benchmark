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
        self.conv2d18 = Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d10 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x52, x48):
        x53=x52.sigmoid()
        x54=operator.mul(x48, x53)
        x55=self.conv2d18(x54)
        x56=self.batchnorm2d10(x55)
        return x56

m = M().eval()
x52 = torch.randn(torch.Size([1, 144, 1, 1]))
x48 = torch.randn(torch.Size([1, 144, 56, 56]))
start = time.time()
output = m(x52, x48)
end = time.time()
print(end-start)
