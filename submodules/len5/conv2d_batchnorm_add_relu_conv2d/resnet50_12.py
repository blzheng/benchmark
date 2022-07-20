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
        self.conv2d33 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)
        self.conv2d34 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x106, x100):
        x107=self.conv2d33(x106)
        x108=self.batchnorm2d33(x107)
        x109=operator.add(x108, x100)
        x110=self.relu28(x109)
        x111=self.conv2d34(x110)
        return x111

m = M().eval()
x106 = torch.randn(torch.Size([1, 256, 14, 14]))
x100 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x106, x100)
end = time.time()
print(end-start)
