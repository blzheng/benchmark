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
        self.batchnorm2d6 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu64 = ReLU6(inplace=True)
        self.conv2d7 = Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
        self.batchnorm2d7 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu65 = ReLU6(inplace=True)
        self.conv2d8 = Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x17):
        x18=self.batchnorm2d6(x17)
        x19=self.relu64(x18)
        x20=self.conv2d7(x19)
        x21=self.batchnorm2d7(x20)
        x22=self.relu65(x21)
        x23=self.conv2d8(x22)
        return x23

m = M().eval()
x17 = torch.randn(torch.Size([1, 144, 56, 56]))
start = time.time()
output = m(x17)
end = time.time()
print(end-start)
