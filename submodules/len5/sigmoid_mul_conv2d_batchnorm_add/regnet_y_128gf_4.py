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
        self.sigmoid4 = Sigmoid()
        self.conv2d27 = Conv2d(1056, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(1056, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x81, x77, x71):
        x82=self.sigmoid4(x81)
        x83=operator.mul(x82, x77)
        x84=self.conv2d27(x83)
        x85=self.batchnorm2d17(x84)
        x86=operator.add(x71, x85)
        return x86

m = M().eval()
x81 = torch.randn(torch.Size([1, 1056, 1, 1]))
x77 = torch.randn(torch.Size([1, 1056, 28, 28]))
x71 = torch.randn(torch.Size([1, 1056, 28, 28]))
start = time.time()
output = m(x81, x77, x71)
end = time.time()
print(end-start)
