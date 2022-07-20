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

    def forward(self, x228, x213, x244):
        x229=operator.add(x228, x213)
        x245=operator.add(x244, x229)
        return x245

m = M().eval()
x228 = torch.randn(torch.Size([1, 112, 14, 14]))
x213 = torch.randn(torch.Size([1, 112, 14, 14]))
x244 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x228, x213, x244)
end = time.time()
print(end-start)
