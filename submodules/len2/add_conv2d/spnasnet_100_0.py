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
        self.conv2d9 = Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x28, x19):
        x29=operator.add(x28, x19)
        x30=self.conv2d9(x29)
        return x30

m = M().eval()
x28 = torch.randn(torch.Size([1, 24, 56, 56]))
x19 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x28, x19)
end = time.time()
print(end-start)
