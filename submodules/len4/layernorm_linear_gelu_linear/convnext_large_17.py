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
        self.layernorm17 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.linear34 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu17 = GELU(approximate='none')
        self.linear35 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x208):
        x209=self.layernorm17(x208)
        x210=self.linear34(x209)
        x211=self.gelu17(x210)
        x212=self.linear35(x211)
        return x212

m = M().eval()
x208 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x208)
end = time.time()
print(end-start)
