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
        self.sigmoid12 = Sigmoid()

    def forward(self, x191, x187):
        x192=self.sigmoid12(x191)
        x193=operator.mul(x192, x187)
        return x193

m = M().eval()
x191 = torch.randn(torch.Size([1, 672, 1, 1]))
x187 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x191, x187)
end = time.time()
print(end-start)
