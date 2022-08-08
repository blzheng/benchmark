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
        self.conv2d61 = Conv2d(336, 84, kernel_size=(1, 1), stride=(1, 1))
        self.relu47 = ReLU()
        self.conv2d62 = Conv2d(84, 336, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid11 = Sigmoid()
        self.conv2d63 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x192, x191, x185):
        x193=self.conv2d61(x192)
        x194=self.relu47(x193)
        x195=self.conv2d62(x194)
        x196=self.sigmoid11(x195)
        x197=operator.mul(x196, x191)
        x198=self.conv2d63(x197)
        x199=self.batchnorm2d39(x198)
        x200=operator.add(x185, x199)
        return x200

m = M().eval()
x192 = torch.randn(torch.Size([1, 336, 1, 1]))
x191 = torch.randn(torch.Size([1, 336, 14, 14]))
x185 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x192, x191, x185)
end = time.time()
print(end-start)
