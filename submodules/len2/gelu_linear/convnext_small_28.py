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
        self.gelu28 = GELU(approximate='none')
        self.linear57 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x331):
        x332=self.gelu28(x331)
        x333=self.linear57(x332)
        return x333

m = M().eval()
x331 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x331)
end = time.time()
print(end-start)
