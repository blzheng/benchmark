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
        self.layernorm32 = LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        self.linear64 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu32 = GELU(approximate='none')
        self.linear65 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x373):
        x374=self.layernorm32(x373)
        x375=self.linear64(x374)
        x376=self.gelu32(x375)
        x377=self.linear65(x376)
        return x377

m = M().eval()
x373 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x373)
end = time.time()
print(end-start)
