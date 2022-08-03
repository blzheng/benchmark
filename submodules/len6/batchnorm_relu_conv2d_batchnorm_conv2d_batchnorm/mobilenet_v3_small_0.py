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
        self.batchnorm2d4 = BatchNorm2d(72, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu3 = ReLU(inplace=True)
        self.conv2d7 = Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(24, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d8 = Conv2d(24, 88, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d6 = BatchNorm2d(88, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x18):
        x19=self.batchnorm2d4(x18)
        x20=self.relu3(x19)
        x21=self.conv2d7(x20)
        x22=self.batchnorm2d5(x21)
        x23=self.conv2d8(x22)
        x24=self.batchnorm2d6(x23)
        return x24

m = M().eval()
x18 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x18)
end = time.time()
print(end-start)
