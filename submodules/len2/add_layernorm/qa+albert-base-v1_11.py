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
        self.layernorm1 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)

    def forward(self, x214, x244):
        x245=operator.add(x214, x244)
        x246=self.layernorm1(x245)
        return x246

m = M().eval()
x214 = torch.randn(torch.Size([1, 384, 768]))
x244 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x214, x244)
end = time.time()
print(end-start)
