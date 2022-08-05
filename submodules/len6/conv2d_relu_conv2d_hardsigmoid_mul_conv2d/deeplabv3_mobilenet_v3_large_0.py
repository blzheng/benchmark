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

    def forward(self, x35, x34):
        x36=self.conv2d11(x35)
        x37=self.relu7(x36)
        x38=self.conv2d12(x37)
        x39=self.hardsigmoid0(x38)
        x40=operator.mul(x39, x34)
        x41=self.conv2d13(x40)
        return x41

m = M().eval()
x35 = torch.randn(torch.Size([1, 72, 1, 1]))
x34 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x35, x34)
end = time.time()
print(end-start)
