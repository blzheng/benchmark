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
        self.relu37 = ReLU(inplace=True)
        self.conv2d43 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu40 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d44 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x139):
        x140=self.relu37(x139)
        x141=self.conv2d43(x140)
        x142=self.batchnorm2d43(x141)
        x143=self.relu40(x142)
        x144=self.conv2d44(x143)
        x145=self.batchnorm2d44(x144)
        x146=self.relu40(x145)
        return x146

m = M().eval()
x139 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x139)
end = time.time()
print(end-start)
