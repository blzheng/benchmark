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
        self.conv2d34 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d22 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)

    def forward(self, x104):
        x105=self.relu24(x104)
        x106=self.conv2d34(x105)
        x107=self.batchnorm2d22(x106)
        x108=self.relu25(x107)
        return x108

m = M().eval()
x104 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x104)
end = time.time()
print(end-start)
