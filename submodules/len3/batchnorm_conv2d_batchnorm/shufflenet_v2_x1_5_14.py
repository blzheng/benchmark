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
        self.batchnorm2d41 = BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d42 = Conv2d(352, 352, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x273):
        x274=self.batchnorm2d41(x273)
        x275=self.conv2d42(x274)
        x276=self.batchnorm2d42(x275)
        return x276

m = M().eval()
x273 = torch.randn(torch.Size([1, 352, 7, 7]))
start = time.time()
output = m(x273)
end = time.time()
print(end-start)
