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
        self.layernorm20 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear18 = Linear(in_features=384, out_features=1536, bias=True)

    def forward(self, x203, x217):
        x218=operator.add(x203, x217)
        x219=self.layernorm20(x218)
        x220=self.linear18(x219)
        return x220

m = M().eval()
x203 = torch.randn(torch.Size([1, 14, 14, 384]))
x217 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x203, x217)
end = time.time()
print(end-start)
