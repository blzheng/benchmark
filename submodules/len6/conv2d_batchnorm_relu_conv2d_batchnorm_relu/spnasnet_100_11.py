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
        self.conv2d33 = Conv2d(80, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU(inplace=True)
        self.conv2d34 = Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=240, bias=False)
        self.batchnorm2d34 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu23 = ReLU(inplace=True)

    def forward(self, x107):
        x108=self.conv2d33(x107)
        x109=self.batchnorm2d33(x108)
        x110=self.relu22(x109)
        x111=self.conv2d34(x110)
        x112=self.batchnorm2d34(x111)
        x113=self.relu23(x112)
        return x113

m = M().eval()
x107 = torch.randn(torch.Size([1, 80, 14, 14]))
start = time.time()
output = m(x107)
end = time.time()
print(end-start)