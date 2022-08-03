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
        self.relu13 = ReLU()
        self.conv2d22 = Conv2d(32, 120, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid2 = Hardsigmoid()
        self.conv2d23 = Conv2d(120, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x63, x61):
        x64=self.relu13(x63)
        x65=self.conv2d22(x64)
        x66=self.hardsigmoid2(x65)
        x67=operator.mul(x66, x61)
        x68=self.conv2d23(x67)
        x69=self.batchnorm2d17(x68)
        return x69

m = M().eval()
x63 = torch.randn(torch.Size([1, 32, 1, 1]))
x61 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x63, x61)
end = time.time()
print(end-start)
