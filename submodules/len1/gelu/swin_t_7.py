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
        self.gelu7 = GELU(approximate=none)

    def forward(self, x197):
        x198=self.gelu7(x197)
        return x198

m = M().eval()
x197 = torch.randn(torch.Size([1, 14, 14, 1536]))
start = time.time()
output = m(x197)
end = time.time()
print(end-start)
