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
        self.linear0 = Linear(in_features=1984, out_features=1000, bias=True)

    def forward(self, x211):
        x212=x211.flatten(1)
        x213=self.linear0(x212)
        return x213

m = M().eval()
x211 = torch.randn(torch.Size([1, 1984, 1, 1]))
start = time.time()
output = m(x211)
end = time.time()
print(end-start)
