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
        self.layernorm2 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)

    def forward(self, x288, x318):
        x319=operator.add(x288, x318)
        x320=self.layernorm1(x319)
        x321=self.linear5(x320)
        x322=torch._C._nn.gelu(x321)
        x323=self.linear6(x322)
        x324=operator.add(x323, x320)
        x325=self.layernorm2(x324)
        return x325

m = M().eval()
x288 = torch.randn(torch.Size([1, 384, 768]))
x318 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x288, x318)
end = time.time()
print(end-start)
