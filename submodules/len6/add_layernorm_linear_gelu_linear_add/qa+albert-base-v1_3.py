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
        self.linear6 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x140, x170):
        x171=operator.add(x140, x170)
        x172=self.layernorm1(x171)
        x173=self.linear5(x172)
        x174=torch._C._nn.gelu(x173)
        x175=self.linear6(x174)
        x176=operator.add(x175, x172)
        return x176

m = M().eval()
x140 = torch.randn(torch.Size([1, 384, 768]))
x170 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x140, x170)
end = time.time()
print(end-start)
