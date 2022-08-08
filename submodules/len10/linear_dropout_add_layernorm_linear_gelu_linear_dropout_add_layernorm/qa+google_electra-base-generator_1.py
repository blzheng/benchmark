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
        self.linear10 = Linear(in_features=256, out_features=256, bias=True)
        self.dropout5 = Dropout(p=0.1, inplace=False)
        self.layernorm3 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear11 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear12 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout6 = Dropout(p=0.1, inplace=False)
        self.layernorm4 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)

    def forward(self, x103, x71):
        x104=self.linear10(x103)
        x105=self.dropout5(x104)
        x106=operator.add(x105, x71)
        x107=self.layernorm3(x106)
        x108=self.linear11(x107)
        x109=torch._C._nn.gelu(x108)
        x110=self.linear12(x109)
        x111=self.dropout6(x110)
        x112=operator.add(x111, x107)
        x113=self.layernorm4(x112)
        return x113

m = M().eval()
x103 = torch.randn(torch.Size([1, 384, 256]))
x71 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x103, x71)
end = time.time()
print(end-start)
