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
        self.relu36 = ReLU(inplace=True)

    def forward(self, x135, x149):
        x150=operator.add(x135, x149)
        x151=self.relu36(x150)
        return x151

m = M().eval()
x135 = torch.randn(torch.Size([1, 1056, 28, 28]))
x149 = torch.randn(torch.Size([1, 1056, 28, 28]))
start = time.time()
output = m(x135, x149)
end = time.time()
print(end-start)
