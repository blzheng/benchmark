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
        self.linear5 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear6 = Linear(in_features=3072, out_features=768, bias=True)
        self.layernorm2 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear1 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x283, x283):
        x284=self.linear5(x283)
        x285=torch._C._nn.gelu(x284)
        x286=self.linear6(x285)
        x287=operator.add(x286, x283)
        x288=self.layernorm2(x287)
        x289=self.linear1(x288)
        return x289

m = M().eval()
x283 = torch.randn(torch.Size([1, 384, 768]))
x283 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x283, x283)
end = time.time()
print(end-start)
