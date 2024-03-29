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
        self.conv2d62 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid8 = Sigmoid()
        self.conv2d63 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x198, x195):
        x199=self.conv2d62(x198)
        x200=self.sigmoid8(x199)
        x201=operator.mul(x200, x195)
        x202=self.conv2d63(x201)
        x203=self.batchnorm2d45(x202)
        return x203

m = M().eval()
x198 = torch.randn(torch.Size([1, 40, 1, 1]))
x195 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x198, x195)
end = time.time()
print(end-start)
