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
        self.linear6 = Linear(in_features=3072, out_features=768, bias=True)
        self.layernorm2 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear73 = Linear(in_features=768, out_features=2, bias=True)

    def forward(self, x469, x468):
        x470=torch._C._nn.gelu(x469)
        x471=self.linear6(x470)
        x472=operator.add(x471, x468)
        x473=self.layernorm2(x472)
        x474=self.linear73(x473)
        return x474

m = M().eval()
x469 = torch.randn(torch.Size([1, 384, 3072]))
x468 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x469, x468)
end = time.time()
print(end-start)
