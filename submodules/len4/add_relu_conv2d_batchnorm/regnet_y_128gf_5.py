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
        self.relu24 = ReLU(inplace=True)
        self.conv2d33 = Conv2d(1056, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(1056, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x87, x101):
        x102=operator.add(x87, x101)
        x103=self.relu24(x102)
        x104=self.conv2d33(x103)
        x105=self.batchnorm2d21(x104)
        return x105

m = M().eval()
x87 = torch.randn(torch.Size([1, 1056, 28, 28]))
x101 = torch.randn(torch.Size([1, 1056, 28, 28]))
start = time.time()
output = m(x87, x101)
end = time.time()
print(end-start)
