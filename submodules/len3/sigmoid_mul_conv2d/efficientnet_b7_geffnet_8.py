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
        self.conv2d41 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x123, x119):
        x124=x123.sigmoid()
        x125=operator.mul(x119, x124)
        x126=self.conv2d41(x125)
        return x126

m = M().eval()
x123 = torch.randn(torch.Size([1, 288, 1, 1]))
x119 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x123, x119)
end = time.time()
print(end-start)
