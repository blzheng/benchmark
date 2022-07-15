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

    def forward(self, x348, x340):
        x349=operator.add(x348, x340)
        return x349

m = M().eval()
x348 = torch.randn(torch.Size([1, 1024, 14, 14]))
x340 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x348, x340)
end = time.time()
print(end-start)
