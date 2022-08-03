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
        self.batchnorm2d18 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu18 = ReLU(inplace=True)
        self.conv2d18 = Conv2d(288, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d19 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu19 = ReLU(inplace=True)
        self.conv2d19 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x66):
        x67=self.batchnorm2d18(x66)
        x68=self.relu18(x67)
        x69=self.conv2d18(x68)
        x70=self.batchnorm2d19(x69)
        x71=self.relu19(x70)
        x72=self.conv2d19(x71)
        return x72

m = M().eval()
x66 = torch.randn(torch.Size([1, 288, 28, 28]))
start = time.time()
output = m(x66)
end = time.time()
print(end-start)
