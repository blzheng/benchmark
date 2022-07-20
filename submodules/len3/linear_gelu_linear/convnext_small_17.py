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
        self.linear34 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu17 = GELU(approximate='none')
        self.linear35 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x209):
        x210=self.linear34(x209)
        x211=self.gelu17(x210)
        x212=self.linear35(x211)
        return x212

m = M().eval()
x209 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x209)
end = time.time()
print(end-start)
