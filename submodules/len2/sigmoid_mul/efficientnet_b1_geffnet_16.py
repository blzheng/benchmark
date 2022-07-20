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

    def forward(self, x244, x240):
        x245=x244.sigmoid()
        x246=operator.mul(x240, x245)
        return x246

m = M().eval()
x244 = torch.randn(torch.Size([1, 672, 1, 1]))
x240 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x244, x240)
end = time.time()
print(end-start)
