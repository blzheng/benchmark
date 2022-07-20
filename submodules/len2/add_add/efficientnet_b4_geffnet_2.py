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

    def forward(self, x219, x205, x234):
        x220=operator.add(x219, x205)
        x235=operator.add(x234, x220)
        return x235

m = M().eval()
x219 = torch.randn(torch.Size([1, 112, 14, 14]))
x205 = torch.randn(torch.Size([1, 112, 14, 14]))
x234 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x219, x205, x234)
end = time.time()
print(end-start)
