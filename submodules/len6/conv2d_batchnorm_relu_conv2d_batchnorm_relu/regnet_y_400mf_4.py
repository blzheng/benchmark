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
        self.conv2d24 = Conv2d(104, 208, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d16 = BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu17 = ReLU(inplace=True)
        self.conv2d25 = Conv2d(208, 208, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=26, bias=False)
        self.batchnorm2d17 = BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu18 = ReLU(inplace=True)

    def forward(self, x71):
        x74=self.conv2d24(x71)
        x75=self.batchnorm2d16(x74)
        x76=self.relu17(x75)
        x77=self.conv2d25(x76)
        x78=self.batchnorm2d17(x77)
        x79=self.relu18(x78)
        return x79

m = M().eval()
x71 = torch.randn(torch.Size([1, 104, 28, 28]))
start = time.time()
output = m(x71)
end = time.time()
print(end-start)
