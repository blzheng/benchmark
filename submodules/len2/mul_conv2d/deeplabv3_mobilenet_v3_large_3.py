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
        self.conv2d40 = Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x118, x113):
        x119=operator.mul(x118, x113)
        x120=self.conv2d40(x119)
        return x120

m = M().eval()
x118 = torch.randn(torch.Size([1, 480, 1, 1]))
x113 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x118, x113)
end = time.time()
print(end-start)
