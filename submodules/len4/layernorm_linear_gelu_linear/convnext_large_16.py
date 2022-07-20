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
        self.layernorm16 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.linear32 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu16 = GELU(approximate='none')
        self.linear33 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x197):
        x198=self.layernorm16(x197)
        x199=self.linear32(x198)
        x200=self.gelu16(x199)
        x201=self.linear33(x200)
        return x201

m = M().eval()
x197 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x197)
end = time.time()
print(end-start)
