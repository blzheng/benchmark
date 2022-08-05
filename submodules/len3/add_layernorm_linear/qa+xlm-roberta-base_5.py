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
        self.layernorm6 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear18 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x152, x148):
        x153=operator.add(x152, x148)
        x154=self.layernorm6(x153)
        x155=self.linear18(x154)
        return x155

m = M().eval()
x152 = torch.randn(torch.Size([1, 384, 768]))
x148 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x152, x148)
end = time.time()
print(end-start)
