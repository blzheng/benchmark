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

    def forward(self, x625, x630):
        x631=operator.mul(x625, x630)
        return x631

m = M().eval()
x625 = torch.randn(torch.Size([1, 2304, 7, 7]))
x630 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x625, x630)
end = time.time()
print(end-start)
