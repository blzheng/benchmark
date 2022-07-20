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
        self.batchnorm2d4 = BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu2 = ReLU(inplace=True)
        self.conv2d5 = Conv2d(64, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x13):
        x14=self.batchnorm2d4(x13)
        x15=self.relu2(x14)
        x16=self.conv2d5(x15)
        return x16

m = M().eval()
x13 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x13)
end = time.time()
print(end-start)