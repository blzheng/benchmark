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

    def forward(self, x396, x391):
        x397=operator.mul(x396, x391)
        return x397

m = M().eval()
x396 = torch.randn(torch.Size([1, 1632, 1, 1]))
x391 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x396, x391)
end = time.time()
print(end-start)
