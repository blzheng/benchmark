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
        self.sigmoid31 = Sigmoid()
        self.conv2d183 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x584, x580):
        x585=self.sigmoid31(x584)
        x586=operator.mul(x585, x580)
        x587=self.conv2d183(x586)
        return x587

m = M().eval()
x584 = torch.randn(torch.Size([1, 1824, 1, 1]))
x580 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x584, x580)
end = time.time()
print(end-start)
