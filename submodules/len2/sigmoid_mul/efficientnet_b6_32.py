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
        self.sigmoid32 = Sigmoid()

    def forward(self, x504, x500):
        x505=self.sigmoid32(x504)
        x506=operator.mul(x505, x500)
        return x506

m = M().eval()
x504 = torch.randn(torch.Size([1, 2064, 1, 1]))
x500 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x504, x500)
end = time.time()
print(end-start)
