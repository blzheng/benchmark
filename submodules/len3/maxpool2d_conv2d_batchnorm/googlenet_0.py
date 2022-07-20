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
        self.maxpool2d0 = MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv2d1 = Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d1 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x16):
        x17=self.maxpool2d0(x16)
        x18=self.conv2d1(x17)
        x19=self.batchnorm2d1(x18)
        return x19

m = M().eval()
x16 = torch.randn(torch.Size([1, 64, 112, 112]))
start = time.time()
output = m(x16)
end = time.time()
print(end-start)