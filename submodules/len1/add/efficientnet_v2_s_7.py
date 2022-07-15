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

    def forward(self, x64, x58):
        x65=operator.add(x64, x58)
        return x65

m = M().eval()
x64 = torch.randn(torch.Size([1, 64, 28, 28]))
x58 = torch.randn(torch.Size([1, 64, 28, 28]))
start = time.time()
output = m(x64, x58)
end = time.time()
print(end-start)
