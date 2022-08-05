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
        self.linear6 = Linear(in_features=3072, out_features=768, bias=True)
        self.layernorm2 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear1 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x210, x209):
        x211=torch._C._nn.gelu(x210)
        x212=self.linear6(x211)
        x213=operator.add(x212, x209)
        x214=self.layernorm2(x213)
        x215=self.linear1(x214)
        return x215

m = M().eval()
x210 = torch.randn(torch.Size([1, 384, 3072]))
x209 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x210, x209)
end = time.time()
print(end-start)
