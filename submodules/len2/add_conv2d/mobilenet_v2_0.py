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
        self.conv2d9 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x16, x24):
        x25=operator.add(x16, x24)
        x26=self.conv2d9(x25)
        return x26

m = M().eval()
x16 = torch.randn(torch.Size([1, 24, 56, 56]))
x24 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x16, x24)
end = time.time()
print(end-start)
