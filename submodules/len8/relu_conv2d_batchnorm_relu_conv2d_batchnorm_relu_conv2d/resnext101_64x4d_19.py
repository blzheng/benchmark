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
        self.relu58 = ReLU(inplace=True)
        self.conv2d64 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d64 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu61 = ReLU(inplace=True)
        self.conv2d65 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        self.batchnorm2d65 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d66 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x209):
        x210=self.relu58(x209)
        x211=self.conv2d64(x210)
        x212=self.batchnorm2d64(x211)
        x213=self.relu61(x212)
        x214=self.conv2d65(x213)
        x215=self.batchnorm2d65(x214)
        x216=self.relu61(x215)
        x217=self.conv2d66(x216)
        return x217

m = M().eval()
x209 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x209)
end = time.time()
print(end-start)
