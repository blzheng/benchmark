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
        self.conv2d41 = Conv2d(488, 488, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=488, bias=False)
        self.batchnorm2d41 = BatchNorm2d(488, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d42 = Conv2d(488, 488, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(488, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x271, x264, x266, x267):
        x272=x271.view(x264, -1, x266, x267)
        x273=self.conv2d41(x272)
        x274=self.batchnorm2d41(x273)
        x275=self.conv2d42(x274)
        x276=self.batchnorm2d42(x275)
        return x276

m = M().eval()
x271 = torch.randn(torch.Size([1, 244, 2, 14, 14]))
x264 = 1
x266 = 14
x267 = 14
start = time.time()
output = m(x271, x264, x266, x267)
end = time.time()
print(end-start)
