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

    def forward(self, x84, x70):
        x85=operator.add(x84, x70)
        return x85

m = M().eval()
x84 = torch.randn(torch.Size([1, 40, 56, 56]))
x70 = torch.randn(torch.Size([1, 40, 56, 56]))
start = time.time()
output = m(x84, x70)
end = time.time()
print(end-start)
