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
        self.relu626 = ReLU6(inplace=True)
        self.conv2d40 = Conv2d(576, 576, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=576, bias=False)
        self.batchnorm2d40 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu627 = ReLU6(inplace=True)
        self.conv2d41 = Conv2d(576, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d42 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x114):
        x115=self.relu626(x114)
        x116=self.conv2d40(x115)
        x117=self.batchnorm2d40(x116)
        x118=self.relu627(x117)
        x119=self.conv2d41(x118)
        x120=self.batchnorm2d41(x119)
        x121=self.conv2d42(x120)
        x122=self.batchnorm2d42(x121)
        return x122

m = M().eval()
x114 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x114)
end = time.time()
print(end-start)
