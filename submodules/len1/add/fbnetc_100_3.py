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

    def forward(self, x60, x51):
        x61=operator.add(x60, x51)
        return x61

m = M().eval()
x60 = torch.randn(torch.Size([1, 32, 28, 28]))
x51 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x60, x51)
end = time.time()
print(end-start)
