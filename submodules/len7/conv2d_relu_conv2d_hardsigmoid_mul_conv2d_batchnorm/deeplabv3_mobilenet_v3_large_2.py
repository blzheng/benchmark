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
        self.conv2d21 = Conv2d(120, 32, kernel_size=(1, 1), stride=(1, 1))
        self.relu13 = ReLU()
        self.conv2d22 = Conv2d(32, 120, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid2 = Hardsigmoid()
        self.conv2d23 = Conv2d(120, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x64, x63):
        x65=self.conv2d21(x64)
        x66=self.relu13(x65)
        x67=self.conv2d22(x66)
        x68=self.hardsigmoid2(x67)
        x69=operator.mul(x68, x63)
        x70=self.conv2d23(x69)
        x71=self.batchnorm2d17(x70)
        return x71

m = M().eval()
x64 = torch.randn(torch.Size([1, 120, 1, 1]))
x63 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x64, x63)
end = time.time()
print(end-start)
