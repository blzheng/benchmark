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
        self.linear5 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear6 = Linear(in_features=3072, out_features=768, bias=True)
        self.layernorm2 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear1 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x97):
        x98=self.layernorm1(x97)
        x99=self.linear5(x98)
        x100=torch._C._nn.gelu(x99)
        x101=self.linear6(x100)
        x102=operator.add(x101, x98)
        x103=self.layernorm2(x102)
        x104=self.linear1(x103)
        return x104

m = M().eval()
x97 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x97)
end = time.time()
print(end-start)