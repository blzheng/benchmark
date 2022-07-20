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

    def forward(self, x26, x14, x39):
        x27=operator.add(x26, x14)
        x40=operator.add(x39, x27)
        return x40

m = M().eval()
x26 = torch.randn(torch.Size([1, 24, 112, 112]))
x14 = torch.randn(torch.Size([1, 24, 112, 112]))
x39 = torch.randn(torch.Size([1, 24, 112, 112]))
start = time.time()
output = m(x26, x14, x39)
end = time.time()
print(end-start)
