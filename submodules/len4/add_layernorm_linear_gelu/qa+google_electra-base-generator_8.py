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
        self.layernorm17 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear53 = Linear(in_features=256, out_features=1024, bias=True)

    def forward(self, x399, x365):
        x400=operator.add(x399, x365)
        x401=self.layernorm17(x400)
        x402=self.linear53(x401)
        x403=torch._C._nn.gelu(x402)
        return x403

m = M().eval()
x399 = torch.randn(torch.Size([1, 384, 256]))
x365 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x399, x365)
end = time.time()
print(end-start)
