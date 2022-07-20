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
        self.sigmoid43 = Sigmoid()
        self.conv2d252 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x809, x805):
        x810=self.sigmoid43(x809)
        x811=operator.mul(x810, x805)
        x812=self.conv2d252(x811)
        return x812

m = M().eval()
x809 = torch.randn(torch.Size([1, 2304, 1, 1]))
x805 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x809, x805)
end = time.time()
print(end-start)
