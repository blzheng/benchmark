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
        self.conv2d5 = Conv2d(8, 64, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid0 = Sigmoid()
        self.conv2d6 = Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x14, x11):
        x15=self.conv2d5(x14)
        x16=self.sigmoid0(x15)
        x17=operator.mul(x16, x11)
        x18=self.conv2d6(x17)
        return x18

m = M().eval()
x14 = torch.randn(torch.Size([1, 8, 1, 1]))
x11 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x14, x11)
end = time.time()
print(end-start)
