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
        self.layernorm3 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear10 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear11 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x104, x70):
        x105=operator.add(x104, x70)
        x106=self.layernorm3(x105)
        x107=self.linear10(x106)
        x108=torch._C._nn.gelu(x107)
        x109=self.linear11(x108)
        return x109

m = M().eval()
x104 = torch.randn(torch.Size([1, 384, 768]))
x70 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x104, x70)
end = time.time()
print(end-start)
