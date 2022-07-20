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
        self.layernorm12 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.linear24 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu12 = GELU(approximate='none')
        self.linear25 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x153):
        x154=self.layernorm12(x153)
        x155=self.linear24(x154)
        x156=self.gelu12(x155)
        x157=self.linear25(x156)
        return x157

m = M().eval()
x153 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x153)
end = time.time()
print(end-start)
