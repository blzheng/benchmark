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
        self.conv2d8 = Conv2d(4, 96, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid1 = Sigmoid()
        self.conv2d9 = Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x23, x20):
        x24=self.conv2d8(x23)
        x25=self.sigmoid1(x24)
        x26=operator.mul(x25, x20)
        x27=self.conv2d9(x26)
        return x27

m = M().eval()
x23 = torch.randn(torch.Size([1, 4, 1, 1]))
x20 = torch.randn(torch.Size([1, 96, 56, 56]))
start = time.time()
output = m(x23, x20)
end = time.time()
print(end-start)