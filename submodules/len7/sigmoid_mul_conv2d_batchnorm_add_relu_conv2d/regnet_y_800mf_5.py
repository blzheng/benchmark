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
        self.sigmoid5 = Sigmoid()
        self.conv2d33 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu24 = ReLU(inplace=True)
        self.conv2d34 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x99, x95, x89):
        x100=self.sigmoid5(x99)
        x101=operator.mul(x100, x95)
        x102=self.conv2d33(x101)
        x103=self.batchnorm2d21(x102)
        x104=operator.add(x89, x103)
        x105=self.relu24(x104)
        x106=self.conv2d34(x105)
        return x106

m = M().eval()
x99 = torch.randn(torch.Size([1, 320, 1, 1]))
x95 = torch.randn(torch.Size([1, 320, 14, 14]))
x89 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x99, x95, x89)
end = time.time()
print(end-start)
