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
        self.linear15 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x101):
        x102=self.linear15(x101)
        return x102

m = M().eval()
x101 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x101)
end = time.time()
print(end-start)
