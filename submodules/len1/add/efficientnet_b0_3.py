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

    def forward(self, x119, x104):
        x120=operator.add(x119, x104)
        return x120

m = M().eval()
x119 = torch.randn(torch.Size([1, 80, 14, 14]))
x104 = torch.randn(torch.Size([1, 80, 14, 14]))
start = time.time()
output = m(x119, x104)
end = time.time()
print(end-start)
