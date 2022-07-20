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
        self.sigmoid5 = Sigmoid()
        self.conv2d33 = Conv2d(208, 208, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x99, x95):
        x100=self.sigmoid5(x99)
        x101=operator.mul(x100, x95)
        x102=self.conv2d33(x101)
        return x102

m = M().eval()
x99 = torch.randn(torch.Size([1, 208, 1, 1]))
x95 = torch.randn(torch.Size([1, 208, 14, 14]))
start = time.time()
output = m(x99, x95)
end = time.time()
print(end-start)