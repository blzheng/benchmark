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
        self.conv2d1 = Conv2d(32, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d1 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = ReLU(inplace=True)
        self.conv2d5 = Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)

    def forward(self, x3, x13):
        x4=self.conv2d1(x3)
        x5=self.batchnorm2d1(x4)
        x14=operator.add(x5, x13)
        x15=self.relu3(x14)
        x16=self.conv2d5(x15)
        return x16

m = M().eval()
x3 = torch.randn(torch.Size([1, 32, 112, 112]))
x13 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x3, x13)
end = time.time()
print(end-start)
