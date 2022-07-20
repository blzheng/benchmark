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
        self.conv2d19 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x56, x42):
        x57=operator.add(x56, x42)
        x58=self.conv2d19(x57)
        return x58

m = M().eval()
x56 = torch.randn(torch.Size([1, 32, 56, 56]))
x42 = torch.randn(torch.Size([1, 32, 56, 56]))
start = time.time()
output = m(x56, x42)
end = time.time()
print(end-start)
