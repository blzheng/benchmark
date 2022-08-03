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
        self.batchnorm2d17 = BatchNorm2d(176, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu11 = ReLU(inplace=True)
        self.conv2d18 = Conv2d(176, 176, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=176, bias=False)
        self.batchnorm2d18 = BatchNorm2d(176, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d19 = Conv2d(176, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d19 = BatchNorm2d(176, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x100):
        x101=self.batchnorm2d17(x100)
        x102=self.relu11(x101)
        x103=self.conv2d18(x102)
        x104=self.batchnorm2d18(x103)
        x105=self.conv2d19(x104)
        x106=self.batchnorm2d19(x105)
        return x106

m = M().eval()
x100 = torch.randn(torch.Size([1, 176, 28, 28]))
start = time.time()
output = m(x100)
end = time.time()
print(end-start)
