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

    def forward(self, x280, x275):
        x281=operator.mul(x280, x275)
        return x281

m = M().eval()
x280 = torch.randn(torch.Size([1, 960, 1, 1]))
x275 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x280, x275)
end = time.time()
print(end-start)
