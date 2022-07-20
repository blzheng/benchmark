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
        self.linear56 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu28 = GELU(approximate='none')

    def forward(self, x330):
        x331=self.linear56(x330)
        x332=self.gelu28(x331)
        return x332

m = M().eval()
x330 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x330)
end = time.time()
print(end-start)
