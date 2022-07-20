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
        self.conv2d113 = Conv2d(1392, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x333, x329):
        x334=x333.sigmoid()
        x335=operator.mul(x329, x334)
        x336=self.conv2d113(x335)
        return x336

m = M().eval()
x333 = torch.randn(torch.Size([1, 1392, 1, 1]))
x329 = torch.randn(torch.Size([1, 1392, 7, 7]))
start = time.time()
output = m(x333, x329)
end = time.time()
print(end-start)
