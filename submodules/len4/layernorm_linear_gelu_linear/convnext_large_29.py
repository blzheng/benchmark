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
        self.layernorm29 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.linear58 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu29 = GELU(approximate='none')
        self.linear59 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x340):
        x341=self.layernorm29(x340)
        x342=self.linear58(x341)
        x343=self.gelu29(x342)
        x344=self.linear59(x343)
        return x344

m = M().eval()
x340 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x340)
end = time.time()
print(end-start)
