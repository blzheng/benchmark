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
        self.layernorm5 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear16 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x147):
        x148=self.layernorm5(x147)
        x149=self.linear16(x148)
        x150=torch._C._nn.gelu(x149)
        return x150

m = M().eval()
x147 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x147)
end = time.time()
print(end-start)
