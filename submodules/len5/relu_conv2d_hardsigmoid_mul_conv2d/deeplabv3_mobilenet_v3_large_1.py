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
        self.relu10 = ReLU()
        self.conv2d17 = Conv2d(32, 120, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid1 = Hardsigmoid()
        self.conv2d18 = Conv2d(120, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x50, x48):
        x51=self.relu10(x50)
        x52=self.conv2d17(x51)
        x53=self.hardsigmoid1(x52)
        x54=operator.mul(x53, x48)
        x55=self.conv2d18(x54)
        return x55

m = M().eval()
x50 = torch.randn(torch.Size([1, 32, 1, 1]))
x48 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x50, x48)
end = time.time()
print(end-start)
