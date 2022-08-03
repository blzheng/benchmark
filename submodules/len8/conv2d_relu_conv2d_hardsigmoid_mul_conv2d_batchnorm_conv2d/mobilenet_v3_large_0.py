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
        self.conv2d11 = Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1))
        self.relu7 = ReLU()
        self.conv2d12 = Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid0 = Hardsigmoid()
        self.conv2d13 = Conv2d(72, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d14 = Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x33, x32):
        x34=self.conv2d11(x33)
        x35=self.relu7(x34)
        x36=self.conv2d12(x35)
        x37=self.hardsigmoid0(x36)
        x38=operator.mul(x37, x32)
        x39=self.conv2d13(x38)
        x40=self.batchnorm2d11(x39)
        x41=self.conv2d14(x40)
        return x41

m = M().eval()
x33 = torch.randn(torch.Size([1, 72, 1, 1]))
x32 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x33, x32)
end = time.time()
print(end-start)
