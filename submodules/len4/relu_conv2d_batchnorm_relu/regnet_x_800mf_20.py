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
        self.relu36 = ReLU(inplace=True)
        self.conv2d41 = Conv2d(672, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu37 = ReLU(inplace=True)

    def forward(self, x130):
        x131=self.relu36(x130)
        x132=self.conv2d41(x131)
        x133=self.batchnorm2d41(x132)
        x134=self.relu37(x133)
        return x134

m = M().eval()
x130 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x130)
end = time.time()
print(end-start)