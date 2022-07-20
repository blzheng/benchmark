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
        self.batchnorm2d2 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = ReLU(inplace=True)
        self.conv2d3 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d3 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x8, x4):
        x9=self.batchnorm2d2(x8)
        x10=operator.add(x9, x4)
        x11=self.relu1(x10)
        x12=self.conv2d3(x11)
        x13=self.batchnorm2d3(x12)
        return x13

m = M().eval()
x8 = torch.randn(torch.Size([1, 64, 56, 56]))
x4 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x8, x4)
end = time.time()
print(end-start)
