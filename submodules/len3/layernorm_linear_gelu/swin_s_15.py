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
        self.layernorm34 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear32 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu15 = GELU(approximate='none')

    def forward(self, x379):
        x380=self.layernorm34(x379)
        x381=self.linear32(x380)
        x382=self.gelu15(x381)
        return x382

m = M().eval()
x379 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x379)
end = time.time()
print(end-start)
