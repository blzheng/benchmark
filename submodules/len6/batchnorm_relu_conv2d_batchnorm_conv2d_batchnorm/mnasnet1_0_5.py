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
        self.batchnorm2d37 = BatchNorm2d(576, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)
        self.conv2d38 = Conv2d(576, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d38 = BatchNorm2d(192, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.conv2d39 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(1152, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x107):
        x108=self.batchnorm2d37(x107)
        x109=self.relu25(x108)
        x110=self.conv2d38(x109)
        x111=self.batchnorm2d38(x110)
        x112=self.conv2d39(x111)
        x113=self.batchnorm2d39(x112)
        return x113

m = M().eval()
x107 = torch.randn(torch.Size([1, 576, 7, 7]))
start = time.time()
output = m(x107)
end = time.time()
print(end-start)
