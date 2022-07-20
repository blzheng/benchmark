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

    def forward(self, x117, x102, x133):
        x118=operator.add(x117, x102)
        x134=operator.add(x133, x118)
        return x134

m = M().eval()
x117 = torch.randn(torch.Size([1, 40, 56, 56]))
x102 = torch.randn(torch.Size([1, 40, 56, 56]))
x133 = torch.randn(torch.Size([1, 40, 56, 56]))
start = time.time()
output = m(x117, x102, x133)
end = time.time()
print(end-start)
