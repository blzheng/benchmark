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
        self.layernorm8 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.linear16 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu8 = GELU(approximate='none')
        self.linear17 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x109):
        x110=self.layernorm8(x109)
        x111=self.linear16(x110)
        x112=self.gelu8(x111)
        x113=self.linear17(x112)
        return x113

m = M().eval()
x109 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x109)
end = time.time()
print(end-start)
