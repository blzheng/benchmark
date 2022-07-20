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
        self.conv2d37 = Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x119, x110):
        x120=operator.add(x119, x110)
        x121=self.conv2d37(x120)
        return x121

m = M().eval()
x119 = torch.randn(torch.Size([1, 64, 14, 14]))
x110 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x119, x110)
end = time.time()
print(end-start)
