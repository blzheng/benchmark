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

    def forward(self, x488, x483):
        x489=operator.mul(x488, x483)
        return x489

m = M().eval()
x488 = torch.randn(torch.Size([1, 1344, 1, 1]))
x483 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x488, x483)
end = time.time()
print(end-start)
