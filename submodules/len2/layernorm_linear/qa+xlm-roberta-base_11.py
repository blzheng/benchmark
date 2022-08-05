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
        self.layernorm12 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear36 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x279):
        x280=self.layernorm12(x279)
        x281=self.linear36(x280)
        return x281

m = M().eval()
x279 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x279)
end = time.time()
print(end-start)
