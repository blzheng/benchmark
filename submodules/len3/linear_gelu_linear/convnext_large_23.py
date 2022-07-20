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
        self.linear46 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu23 = GELU(approximate='none')
        self.linear47 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x275):
        x276=self.linear46(x275)
        x277=self.gelu23(x276)
        x278=self.linear47(x277)
        return x278

m = M().eval()
x275 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x275)
end = time.time()
print(end-start)
