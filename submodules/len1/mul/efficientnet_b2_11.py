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

    def forward(self, x176, x171):
        x177=operator.mul(x176, x171)
        return x177

m = M().eval()
x176 = torch.randn(torch.Size([1, 528, 1, 1]))
x171 = torch.randn(torch.Size([1, 528, 14, 14]))
start = time.time()
output = m(x176, x171)
end = time.time()
print(end-start)
