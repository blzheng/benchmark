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
        self.batchnorm2d41 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu41 = ReLU(inplace=True)
        self.conv2d41 = Conv2d(432, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu42 = ReLU(inplace=True)

    def forward(self, x148):
        x149=self.batchnorm2d41(x148)
        x150=self.relu41(x149)
        x151=self.conv2d41(x150)
        x152=self.batchnorm2d42(x151)
        x153=self.relu42(x152)
        return x153

m = M().eval()
x148 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x148)
end = time.time()
print(end-start)
