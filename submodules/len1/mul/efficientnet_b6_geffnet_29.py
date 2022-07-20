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

    def forward(self, x433, x438):
        x439=operator.mul(x433, x438)
        return x439

m = M().eval()
x433 = torch.randn(torch.Size([1, 1200, 14, 14]))
x438 = torch.randn(torch.Size([1, 1200, 1, 1]))
start = time.time()
output = m(x433, x438)
end = time.time()
print(end-start)