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
        self.layernorm23 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear70 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x525):
        x526=self.layernorm23(x525)
        x527=self.linear70(x526)
        x528=torch._C._nn.gelu(x527)
        return x528

m = M().eval()
x525 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x525)
end = time.time()
print(end-start)
