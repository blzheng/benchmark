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

    def forward(self, x219, x214):
        x220=operator.mul(x219, x214)
        return x220

m = M().eval()
x219 = torch.randn(torch.Size([1, 1056, 1, 1]))
x214 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x219, x214)
end = time.time()
print(end-start)
