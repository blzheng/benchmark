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
        self.conv2d41 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d41 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu37 = ReLU(inplace=True)
        self.conv2d42 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x133):
        x134=self.conv2d41(x133)
        x135=self.batchnorm2d41(x134)
        x136=self.relu37(x135)
        x137=self.conv2d42(x136)
        x138=self.batchnorm2d42(x137)
        return x138

m = M().eval()
x133 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x133)
end = time.time()
print(end-start)
