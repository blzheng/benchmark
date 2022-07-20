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
        self.batchnorm2d32 = BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu21 = ReLU(inplace=True)
        self.conv2d33 = Conv2d(116, 116, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=116, bias=False)
        self.batchnorm2d33 = BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d34 = Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x210):
        x211=self.batchnorm2d32(x210)
        x212=self.relu21(x211)
        x213=self.conv2d33(x212)
        x214=self.batchnorm2d33(x213)
        x215=self.conv2d34(x214)
        return x215

m = M().eval()
x210 = torch.randn(torch.Size([1, 116, 14, 14]))
start = time.time()
output = m(x210)
end = time.time()
print(end-start)
