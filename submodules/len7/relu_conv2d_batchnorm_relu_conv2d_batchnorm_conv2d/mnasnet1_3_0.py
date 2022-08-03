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
        self.relu0 = ReLU(inplace=True)
        self.conv2d1 = Conv2d(40, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
        self.batchnorm2d1 = BatchNorm2d(40, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu1 = ReLU(inplace=True)
        self.conv2d2 = Conv2d(40, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d2 = BatchNorm2d(24, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.conv2d3 = Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x2):
        x3=self.relu0(x2)
        x4=self.conv2d1(x3)
        x5=self.batchnorm2d1(x4)
        x6=self.relu1(x5)
        x7=self.conv2d2(x6)
        x8=self.batchnorm2d2(x7)
        x9=self.conv2d3(x8)
        return x9

m = M().eval()
x2 = torch.randn(torch.Size([1, 40, 112, 112]))
start = time.time()
output = m(x2)
end = time.time()
print(end-start)
