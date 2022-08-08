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
        self.relu27 = ReLU(inplace=True)
        self.conv2d31 = Conv2d(288, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)
        self.conv2d32 = Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=18, bias=False)
        self.batchnorm2d32 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu29 = ReLU(inplace=True)
        self.conv2d33 = Conv2d(288, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x98):
        x99=self.relu27(x98)
        x100=self.conv2d31(x99)
        x101=self.batchnorm2d31(x100)
        x102=self.relu28(x101)
        x103=self.conv2d32(x102)
        x104=self.batchnorm2d32(x103)
        x105=self.relu29(x104)
        x106=self.conv2d33(x105)
        x107=self.batchnorm2d33(x106)
        return x107

m = M().eval()
x98 = torch.randn(torch.Size([1, 288, 14, 14]))
start = time.time()
output = m(x98)
end = time.time()
print(end-start)
