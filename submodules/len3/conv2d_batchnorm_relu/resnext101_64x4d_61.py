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
        self.conv2d94 = Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d94 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu91 = ReLU(inplace=True)

    def forward(self, x310):
        x311=self.conv2d94(x310)
        x312=self.batchnorm2d94(x311)
        x313=self.relu91(x312)
        return x313

m = M().eval()
x310 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x310)
end = time.time()
print(end-start)
