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
        self.conv2d89 = Conv2d(176, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x287, x272):
        x288=operator.add(x287, x272)
        x289=self.conv2d89(x288)
        return x289

m = M().eval()
x287 = torch.randn(torch.Size([1, 176, 14, 14]))
x272 = torch.randn(torch.Size([1, 176, 14, 14]))
start = time.time()
output = m(x287, x272)
end = time.time()
print(end-start)
