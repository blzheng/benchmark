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
        self.layernorm1 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear4 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear5 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout3 = Dropout(p=0.1, inplace=False)
        self.layernorm2 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)

    def forward(self, x63):
        x64=self.layernorm1(x63)
        x65=self.linear4(x64)
        x66=torch._C._nn.gelu(x65)
        x67=self.linear5(x66)
        x68=self.dropout3(x67)
        x69=operator.add(x68, x64)
        x70=self.layernorm2(x69)
        return x70

m = M().eval()
x63 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x63)
end = time.time()
print(end-start)
