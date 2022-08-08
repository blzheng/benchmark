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
        self.relu4 = ReLU(inplace=True)
        self.conv2d8 = Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(24, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x24):
        x25=self.relu4(x24)
        x26=self.conv2d8(x25)
        x27=self.batchnorm2d8(x26)
        return x27

m = M().eval()
x24 = torch.randn(torch.Size([1, 72, 56, 56]))
start = time.time()
output = m(x24)
end = time.time()
print(end-start)
