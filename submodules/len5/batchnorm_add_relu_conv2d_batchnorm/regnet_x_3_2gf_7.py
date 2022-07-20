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
        self.batchnorm2d20 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu18 = ReLU(inplace=True)
        self.conv2d21 = Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x64, x57):
        x65=self.batchnorm2d20(x64)
        x66=operator.add(x57, x65)
        x67=self.relu18(x66)
        x68=self.conv2d21(x67)
        x69=self.batchnorm2d21(x68)
        return x69

m = M().eval()
x64 = torch.randn(torch.Size([1, 192, 28, 28]))
x57 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x64, x57)
end = time.time()
print(end-start)
