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

    def forward(self, x333, x329):
        x334=x333.sigmoid()
        x335=operator.mul(x329, x334)
        return x335

m = M().eval()
x333 = torch.randn(torch.Size([1, 1392, 1, 1]))
x329 = torch.randn(torch.Size([1, 1392, 7, 7]))
start = time.time()
output = m(x333, x329)
end = time.time()
print(end-start)
