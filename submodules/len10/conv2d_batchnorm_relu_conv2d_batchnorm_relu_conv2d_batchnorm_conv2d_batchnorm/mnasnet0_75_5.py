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
        self.conv2d36 = Conv2d(72, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(432, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu24 = ReLU(inplace=True)
        self.conv2d37 = Conv2d(432, 432, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=432, bias=False)
        self.batchnorm2d37 = BatchNorm2d(432, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)
        self.conv2d38 = Conv2d(432, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d38 = BatchNorm2d(144, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.conv2d39 = Conv2d(144, 864, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(864, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x103):
        x104=self.conv2d36(x103)
        x105=self.batchnorm2d36(x104)
        x106=self.relu24(x105)
        x107=self.conv2d37(x106)
        x108=self.batchnorm2d37(x107)
        x109=self.relu25(x108)
        x110=self.conv2d38(x109)
        x111=self.batchnorm2d38(x110)
        x112=self.conv2d39(x111)
        x113=self.batchnorm2d39(x112)
        return x113

m = M().eval()
x103 = torch.randn(torch.Size([1, 72, 14, 14]))
start = time.time()
output = m(x103)
end = time.time()
print(end-start)
