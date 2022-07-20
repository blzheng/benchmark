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
        self.conv2d12 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x38, x29):
        x39=operator.add(x38, x29)
        x40=self.conv2d12(x39)
        return x40

m = M().eval()
x38 = torch.randn(torch.Size([1, 24, 56, 56]))
x29 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x38, x29)
end = time.time()
print(end-start)
