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
        self.linear46 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu23 = GELU(approximate='none')

    def forward(self, x275):
        x276=self.linear46(x275)
        x277=self.gelu23(x276)
        return x277

m = M().eval()
x275 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x275)
end = time.time()
print(end-start)
