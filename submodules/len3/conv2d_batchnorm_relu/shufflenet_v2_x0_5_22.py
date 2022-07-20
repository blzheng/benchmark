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
        self.conv2d34 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d34 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU(inplace=True)

    def forward(self, x214):
        x215=self.conv2d34(x214)
        x216=self.batchnorm2d34(x215)
        x217=self.relu22(x216)
        return x217

m = M().eval()
x214 = torch.randn(torch.Size([1, 48, 14, 14]))
start = time.time()
output = m(x214)
end = time.time()
print(end-start)
