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

    def forward(self, x503, x499):
        x504=self.sigmoid32(x503)
        x505=operator.mul(x504, x499)
        return x505

m = M().eval()
x503 = torch.randn(torch.Size([1, 1344, 1, 1]))
x499 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x503, x499)
end = time.time()
print(end-start)
