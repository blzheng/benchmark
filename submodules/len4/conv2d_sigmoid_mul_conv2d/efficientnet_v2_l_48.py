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
        self.conv2d276 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid48 = Sigmoid()
        self.conv2d277 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x888, x885):
        x889=self.conv2d276(x888)
        x890=self.sigmoid48(x889)
        x891=operator.mul(x890, x885)
        x892=self.conv2d277(x891)
        return x892

m = M().eval()
x888 = torch.randn(torch.Size([1, 96, 1, 1]))
x885 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x888, x885)
end = time.time()
print(end-start)
