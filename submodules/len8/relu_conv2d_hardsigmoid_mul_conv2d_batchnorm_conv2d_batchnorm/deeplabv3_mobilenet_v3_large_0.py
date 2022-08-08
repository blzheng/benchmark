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
        self.relu7 = ReLU()
        self.conv2d12 = Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid0 = Hardsigmoid()
        self.conv2d13 = Conv2d(72, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d14 = Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(120, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x36, x34):
        x37=self.relu7(x36)
        x38=self.conv2d12(x37)
        x39=self.hardsigmoid0(x38)
        x40=operator.mul(x39, x34)
        x41=self.conv2d13(x40)
        x42=self.batchnorm2d11(x41)
        x43=self.conv2d14(x42)
        x44=self.batchnorm2d12(x43)
        return x44

m = M().eval()
x36 = torch.randn(torch.Size([1, 24, 1, 1]))
x34 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x36, x34)
end = time.time()
print(end-start)
