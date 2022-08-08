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
        self.conv2d5 = Conv2d(8, 48, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid0 = Sigmoid()
        self.conv2d6 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d4 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = ReLU(inplace=True)
        self.conv2d7 = Conv2d(48, 104, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d5 = BatchNorm2d(104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x14, x11, x5):
        x15=self.conv2d5(x14)
        x16=self.sigmoid0(x15)
        x17=operator.mul(x16, x11)
        x18=self.conv2d6(x17)
        x19=self.batchnorm2d4(x18)
        x20=operator.add(x5, x19)
        x21=self.relu4(x20)
        x22=self.conv2d7(x21)
        x23=self.batchnorm2d5(x22)
        return x23

m = M().eval()
x14 = torch.randn(torch.Size([1, 8, 1, 1]))
x11 = torch.randn(torch.Size([1, 48, 56, 56]))
x5 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x14, x11, x5)
end = time.time()
print(end-start)
