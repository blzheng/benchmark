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
        self.layernorm7 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear23 = Linear(in_features=256, out_features=1024, bias=True)

    def forward(self, x189, x155):
        x190=operator.add(x189, x155)
        x191=self.layernorm7(x190)
        x192=self.linear23(x191)
        x193=torch._C._nn.gelu(x192)
        return x193

m = M().eval()
x189 = torch.randn(torch.Size([1, 384, 256]))
x155 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x189, x155)
end = time.time()
print(end-start)
