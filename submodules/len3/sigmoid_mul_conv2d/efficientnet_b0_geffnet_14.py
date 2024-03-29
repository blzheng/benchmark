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
        self.conv2d74 = Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x215, x211):
        x216=x215.sigmoid()
        x217=operator.mul(x211, x216)
        x218=self.conv2d74(x217)
        return x218

m = M().eval()
x215 = torch.randn(torch.Size([1, 1152, 1, 1]))
x211 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x215, x211)
end = time.time()
print(end-start)
