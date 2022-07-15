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

    def forward(self, x137, x142):
        x143=operator.mul(x137, x142)
        return x143

m = M().eval()
x137 = torch.randn(torch.Size([1, 672, 14, 14]))
x142 = torch.randn(torch.Size([1, 672, 1, 1]))
start = time.time()
output = m(x137, x142)
end = time.time()
print(end-start)
