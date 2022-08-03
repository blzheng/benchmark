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
        self.conv2d39 = Conv2d(248, 1488, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(1488, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu26 = ReLU(inplace=True)
        self.conv2d40 = Conv2d(1488, 1488, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1488, bias=False)
        self.batchnorm2d40 = BatchNorm2d(1488, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu27 = ReLU(inplace=True)
        self.conv2d41 = Conv2d(1488, 248, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(248, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x111):
        x112=self.conv2d39(x111)
        x113=self.batchnorm2d39(x112)
        x114=self.relu26(x113)
        x115=self.conv2d40(x114)
        x116=self.batchnorm2d40(x115)
        x117=self.relu27(x116)
        x118=self.conv2d41(x117)
        x119=self.batchnorm2d41(x118)
        return x119

m = M().eval()
x111 = torch.randn(torch.Size([1, 248, 7, 7]))
start = time.time()
output = m(x111)
end = time.time()
print(end-start)
