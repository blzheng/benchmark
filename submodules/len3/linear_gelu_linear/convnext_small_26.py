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
        self.linear52 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu26 = GELU(approximate='none')
        self.linear53 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x308):
        x309=self.linear52(x308)
        x310=self.gelu26(x309)
        x311=self.linear53(x310)
        return x311

m = M().eval()
x308 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x308)
end = time.time()
print(end-start)
