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
        self.hardsigmoid0 = Hardsigmoid()
        self.conv2d13 = Conv2d(72, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x38, x34):
        x39=self.hardsigmoid0(x38)
        x40=operator.mul(x39, x34)
        x41=self.conv2d13(x40)
        return x41

m = M().eval()
x38 = torch.randn(torch.Size([1, 72, 1, 1]))
x34 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x38, x34)
end = time.time()
print(end-start)
