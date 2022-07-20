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
        self.conv2d28 = Conv2d(144, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x84, x79):
        x85=operator.mul(x84, x79)
        x86=self.conv2d28(x85)
        return x86

m = M().eval()
x84 = torch.randn(torch.Size([1, 144, 1, 1]))
x79 = torch.randn(torch.Size([1, 144, 28, 28]))
start = time.time()
output = m(x84, x79)
end = time.time()
print(end-start)
