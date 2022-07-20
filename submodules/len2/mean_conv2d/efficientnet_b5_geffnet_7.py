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
        self.conv2d35 = Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x106):
        x107=x106.mean((2, 3),keepdim=True)
        x108=self.conv2d35(x107)
        return x108

m = M().eval()
x106 = torch.randn(torch.Size([1, 240, 56, 56]))
start = time.time()
output = m(x106)
end = time.time()
print(end-start)
