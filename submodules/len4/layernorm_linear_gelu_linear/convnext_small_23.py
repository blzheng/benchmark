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
        self.layernorm23 = LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        self.linear46 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu23 = GELU(approximate='none')
        self.linear47 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x274):
        x275=self.layernorm23(x274)
        x276=self.linear46(x275)
        x277=self.gelu23(x276)
        x278=self.linear47(x277)
        return x278

m = M().eval()
x274 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x274)
end = time.time()
print(end-start)
