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
        self.dropout2 = Dropout(p=0.1, inplace=False)
        self.layernorm1 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear5 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear6 = Linear(in_features=3072, out_features=768, bias=True)
        self.layernorm2 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)

    def forward(self, x280, x251):
        x281=self.dropout2(x280)
        x282=operator.add(x251, x281)
        x283=self.layernorm1(x282)
        x284=self.linear5(x283)
        x285=torch._C._nn.gelu(x284)
        x286=self.linear6(x285)
        x287=operator.add(x286, x283)
        x288=self.layernorm2(x287)
        return x288

m = M().eval()
x280 = torch.randn(torch.Size([1, 384, 768]))
x251 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x280, x251)
end = time.time()
print(end-start)
