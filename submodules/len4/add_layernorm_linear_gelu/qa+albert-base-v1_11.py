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
        self.layernorm1 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear5 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x436, x466):
        x467=operator.add(x436, x466)
        x468=self.layernorm1(x467)
        x469=self.linear5(x468)
        x470=torch._C._nn.gelu(x469)
        return x470

m = M().eval()
x436 = torch.randn(torch.Size([1, 384, 768]))
x466 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x436, x466)
end = time.time()
print(end-start)
