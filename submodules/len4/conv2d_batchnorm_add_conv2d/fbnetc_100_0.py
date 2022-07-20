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
        self.conv2d3 = Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d3 = BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d4 = Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x10, x3):
        x11=self.conv2d3(x10)
        x12=self.batchnorm2d3(x11)
        x13=operator.add(x12, x3)
        x14=self.conv2d4(x13)
        return x14

m = M().eval()
x10 = torch.randn(torch.Size([1, 16, 112, 112]))
x3 = torch.randn(torch.Size([1, 16, 112, 112]))
start = time.time()
output = m(x10, x3)
end = time.time()
print(end-start)
