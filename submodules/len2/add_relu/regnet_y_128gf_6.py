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
        self.relu28 = ReLU(inplace=True)

    def forward(self, x103, x117):
        x118=operator.add(x103, x117)
        x119=self.relu28(x118)
        return x119

m = M().eval()
x103 = torch.randn(torch.Size([1, 1056, 28, 28]))
x117 = torch.randn(torch.Size([1, 1056, 28, 28]))
start = time.time()
output = m(x103, x117)
end = time.time()
print(end-start)
