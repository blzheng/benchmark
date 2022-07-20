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

    def forward(self, x102, x87, x118):
        x103=operator.add(x102, x87)
        x119=operator.add(x118, x103)
        return x119

m = M().eval()
x102 = torch.randn(torch.Size([1, 48, 28, 28]))
x87 = torch.randn(torch.Size([1, 48, 28, 28]))
x118 = torch.randn(torch.Size([1, 48, 28, 28]))
start = time.time()
output = m(x102, x87, x118)
end = time.time()
print(end-start)
