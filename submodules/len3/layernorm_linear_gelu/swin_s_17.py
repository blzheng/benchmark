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
        self.layernorm38 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear36 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu17 = GELU(approximate='none')

    def forward(self, x425):
        x426=self.layernorm38(x425)
        x427=self.linear36(x426)
        x428=self.gelu17(x427)
        return x428

m = M().eval()
x425 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x425)
end = time.time()
print(end-start)
