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

    def forward(self, x100, x98):
        x101=self.linear6(x100)
        x102=operator.add(x101, x98)
        x103=self.layernorm2(x102)
        return x103

m = M().eval()
x100 = torch.randn(torch.Size([1, 384, 3072]))
x98 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x100, x98)
end = time.time()
print(end-start)
