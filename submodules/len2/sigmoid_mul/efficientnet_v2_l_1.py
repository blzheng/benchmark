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
        self.sigmoid1 = Sigmoid()

    def forward(self, x141, x137):
        x142=self.sigmoid1(x141)
        x143=operator.mul(x142, x137)
        return x143

m = M().eval()
x141 = torch.randn(torch.Size([1, 768, 1, 1]))
x137 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x141, x137)
end = time.time()
print(end-start)
