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
        self.conv2d63 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu48 = ReLU(inplace=True)
        self.conv2d64 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x196, x191, x185):
        x197=operator.mul(x196, x191)
        x198=self.conv2d63(x197)
        x199=self.batchnorm2d39(x198)
        x200=operator.add(x185, x199)
        x201=self.relu48(x200)
        x202=self.conv2d64(x201)
        return x202

m = M().eval()
x196 = torch.randn(torch.Size([1, 336, 1, 1]))
x191 = torch.randn(torch.Size([1, 336, 14, 14]))
x185 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x196, x191, x185)
end = time.time()
print(end-start)
