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

    def forward(self, x103, x88, x119):
        x104=operator.add(x103, x88)
        x120=operator.add(x119, x104)
        return x120

m = M().eval()
x103 = torch.randn(torch.Size([1, 80, 14, 14]))
x88 = torch.randn(torch.Size([1, 80, 14, 14]))
x119 = torch.randn(torch.Size([1, 80, 14, 14]))
start = time.time()
output = m(x103, x88, x119)
end = time.time()
print(end-start)
