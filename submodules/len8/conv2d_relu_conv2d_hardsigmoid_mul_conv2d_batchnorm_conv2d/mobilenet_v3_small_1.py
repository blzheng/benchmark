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
        self.conv2d13 = Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1))
        self.relu6 = ReLU()
        self.conv2d14 = Conv2d(24, 96, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid1 = Hardsigmoid()
        self.conv2d15 = Conv2d(96, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d16 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x38, x37):
        x39=self.conv2d13(x38)
        x40=self.relu6(x39)
        x41=self.conv2d14(x40)
        x42=self.hardsigmoid1(x41)
        x43=operator.mul(x42, x37)
        x44=self.conv2d15(x43)
        x45=self.batchnorm2d11(x44)
        x46=self.conv2d16(x45)
        return x46

m = M().eval()
x38 = torch.randn(torch.Size([1, 96, 1, 1]))
x37 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x38, x37)
end = time.time()
print(end-start)
