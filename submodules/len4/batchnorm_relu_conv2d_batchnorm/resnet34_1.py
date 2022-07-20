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
        self.batchnorm2d3 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = ReLU(inplace=True)
        self.conv2d4 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d4 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x12):
        x13=self.batchnorm2d3(x12)
        x14=self.relu3(x13)
        x15=self.conv2d4(x14)
        x16=self.batchnorm2d4(x15)
        return x16

m = M().eval()
x12 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x12)
end = time.time()
print(end-start)
