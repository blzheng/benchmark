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

    def forward(self, x337, x322, x353):
        x338=operator.add(x337, x322)
        x354=operator.add(x353, x338)
        return x354

m = M().eval()
x337 = torch.randn(torch.Size([1, 144, 14, 14]))
x322 = torch.randn(torch.Size([1, 144, 14, 14]))
x353 = torch.randn(torch.Size([1, 144, 14, 14]))
start = time.time()
output = m(x337, x322, x353)
end = time.time()
print(end-start)
