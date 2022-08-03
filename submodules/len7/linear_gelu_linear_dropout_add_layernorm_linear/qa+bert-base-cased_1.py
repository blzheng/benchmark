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
        self.linear10 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear11 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout6 = Dropout(p=0.1, inplace=False)
        self.layernorm4 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear12 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x106, x106):
        x107=self.linear10(x106)
        x108=torch._C._nn.gelu(x107)
        x109=self.linear11(x108)
        x110=self.dropout6(x109)
        x111=operator.add(x110, x106)
        x112=self.layernorm4(x111)
        x113=self.linear12(x112)
        return x113

m = M().eval()
x106 = torch.randn(torch.Size([1, 384, 768]))
x106 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x106, x106)
end = time.time()
print(end-start)
