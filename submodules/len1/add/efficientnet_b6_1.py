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

    def forward(self, x39, x27):
        x40=operator.add(x39, x27)
        return x40

m = M().eval()
x39 = torch.randn(torch.Size([1, 32, 112, 112]))
x27 = torch.randn(torch.Size([1, 32, 112, 112]))
start = time.time()
output = m(x39, x27)
end = time.time()
print(end-start)
