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

    def forward(self, x26, x40, x48):
        x41=operator.add(x26, x40)
        x49=operator.add(x41, x48)
        return x49

m = M().eval()
x26 = torch.randn(torch.Size([1, 56, 56, 96]))
x40 = torch.randn(torch.Size([1, 56, 56, 96]))
x48 = torch.randn(torch.Size([1, 56, 56, 96]))
start = time.time()
output = m(x26, x40, x48)
end = time.time()
print(end-start)
