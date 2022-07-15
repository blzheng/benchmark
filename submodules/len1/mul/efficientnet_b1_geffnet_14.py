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

    def forward(self, x210, x215):
        x216=operator.mul(x210, x215)
        return x216

m = M().eval()
x210 = torch.randn(torch.Size([1, 672, 14, 14]))
x215 = torch.randn(torch.Size([1, 672, 1, 1]))
start = time.time()
output = m(x210, x215)
end = time.time()
print(end-start)
