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
        self.batchnorm2d4 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu63 = ReLU6(inplace=True)
        self.conv2d5 = Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x12):
        x13=self.batchnorm2d4(x12)
        x14=self.relu63(x13)
        x15=self.conv2d5(x14)
        x16=self.batchnorm2d5(x15)
        return x16

m = M().eval()
x12 = torch.randn(torch.Size([1, 96, 56, 56]))
start = time.time()
output = m(x12)
end = time.time()
print(end-start)
