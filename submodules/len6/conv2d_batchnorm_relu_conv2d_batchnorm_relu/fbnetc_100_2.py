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
        self.conv2d4 = Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d4 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = ReLU(inplace=True)
        self.conv2d5 = Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
        self.batchnorm2d5 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = ReLU(inplace=True)

    def forward(self, x13):
        x14=self.conv2d4(x13)
        x15=self.batchnorm2d4(x14)
        x16=self.relu3(x15)
        x17=self.conv2d5(x16)
        x18=self.batchnorm2d5(x17)
        x19=self.relu4(x18)
        return x19

m = M().eval()
x13 = torch.randn(torch.Size([1, 16, 112, 112]))
start = time.time()
output = m(x13)
end = time.time()
print(end-start)
