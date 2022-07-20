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

    def forward(self, x322, x308, x337):
        x323=operator.add(x322, x308)
        x338=operator.add(x337, x323)
        return x338

m = M().eval()
x322 = torch.randn(torch.Size([1, 144, 14, 14]))
x308 = torch.randn(torch.Size([1, 144, 14, 14]))
x337 = torch.randn(torch.Size([1, 144, 14, 14]))
start = time.time()
output = m(x322, x308, x337)
end = time.time()
print(end-start)
