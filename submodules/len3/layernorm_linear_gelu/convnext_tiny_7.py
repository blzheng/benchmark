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
        self.layernorm7 = LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        self.linear14 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu7 = GELU(approximate='none')

    def forward(self, x98):
        x99=self.layernorm7(x98)
        x100=self.linear14(x99)
        x101=self.gelu7(x100)
        return x101

m = M().eval()
x98 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x98)
end = time.time()
print(end-start)
