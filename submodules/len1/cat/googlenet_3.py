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

    def forward(self, x68, x74, x80, x84):
        x85=torch.cat([x68, x74, x80, x84], 1)
        return x85

m = M().eval()
x68 = torch.randn(torch.Size([1, 192, 14, 14]))
x74 = torch.randn(torch.Size([1, 208, 14, 14]))
x80 = torch.randn(torch.Size([1, 48, 14, 14]))
x84 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x68, x74, x80, x84)
end = time.time()
print(end-start)
