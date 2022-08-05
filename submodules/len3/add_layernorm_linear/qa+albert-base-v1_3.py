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
        self.layernorm2 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear1 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x101, x98):
        x102=operator.add(x101, x98)
        x103=self.layernorm2(x102)
        x104=self.linear1(x103)
        return x104

m = M().eval()
x101 = torch.randn(torch.Size([1, 384, 768]))
x98 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x101, x98)
end = time.time()
print(end-start)
