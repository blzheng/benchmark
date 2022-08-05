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
        self.layernorm11 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear34 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x273):
        x274=self.layernorm11(x273)
        x275=self.linear34(x274)
        x276=torch._C._nn.gelu(x275)
        return x276

m = M().eval()
x273 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x273)
end = time.time()
print(end-start)
