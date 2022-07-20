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

    def forward(self, x40, x28, x53):
        x41=operator.add(x40, x28)
        x54=operator.add(x53, x41)
        return x54

m = M().eval()
x40 = torch.randn(torch.Size([1, 32, 112, 112]))
x28 = torch.randn(torch.Size([1, 32, 112, 112]))
x53 = torch.randn(torch.Size([1, 32, 112, 112]))
start = time.time()
output = m(x40, x28, x53)
end = time.time()
print(end-start)
