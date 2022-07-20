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
        self.batchnorm2d1 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = ReLU(inplace=True)
        self.conv2d7 = Conv2d(48, 104, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d5 = BatchNorm2d(104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x4, x19):
        x5=self.batchnorm2d1(x4)
        x20=operator.add(x5, x19)
        x21=self.relu4(x20)
        x22=self.conv2d7(x21)
        x23=self.batchnorm2d5(x22)
        return x23

m = M().eval()
x4 = torch.randn(torch.Size([1, 48, 56, 56]))
x19 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x4, x19)
end = time.time()
print(end-start)
