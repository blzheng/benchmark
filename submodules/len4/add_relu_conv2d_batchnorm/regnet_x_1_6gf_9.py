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
        self.relu30 = ReLU(inplace=True)
        self.conv2d34 = Conv2d(408, 408, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d34 = BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x99, x107):
        x108=operator.add(x99, x107)
        x109=self.relu30(x108)
        x110=self.conv2d34(x109)
        x111=self.batchnorm2d34(x110)
        return x111

m = M().eval()
x99 = torch.randn(torch.Size([1, 408, 14, 14]))
x107 = torch.randn(torch.Size([1, 408, 14, 14]))
start = time.time()
output = m(x99, x107)
end = time.time()
print(end-start)
