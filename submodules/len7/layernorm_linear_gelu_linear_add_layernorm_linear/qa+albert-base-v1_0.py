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

    def forward(self, x60):
        x61=self.layernorm1(x60)
        x62=self.linear5(x61)
        x63=torch._C._nn.gelu(x62)
        x64=self.linear6(x63)
        x65=operator.add(x64, x61)
        x66=self.layernorm2(x65)
        x67=self.linear1(x66)
        return x67

m = M().eval()
x60 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x60)
end = time.time()
print(end-start)
