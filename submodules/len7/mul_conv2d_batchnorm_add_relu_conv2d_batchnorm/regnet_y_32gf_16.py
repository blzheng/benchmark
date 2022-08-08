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
        self.conv2d88 = Conv2d(1392, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d54 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu68 = ReLU(inplace=True)
        self.conv2d89 = Conv2d(1392, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x276, x271, x265):
        x277=operator.mul(x276, x271)
        x278=self.conv2d88(x277)
        x279=self.batchnorm2d54(x278)
        x280=operator.add(x265, x279)
        x281=self.relu68(x280)
        x282=self.conv2d89(x281)
        x283=self.batchnorm2d55(x282)
        return x283

m = M().eval()
x276 = torch.randn(torch.Size([1, 1392, 1, 1]))
x271 = torch.randn(torch.Size([1, 1392, 14, 14]))
x265 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x276, x271, x265)
end = time.time()
print(end-start)
