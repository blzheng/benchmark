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
        self.layernorm25 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.linear50 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu25 = GELU(approximate='none')
        self.linear51 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x296):
        x297=self.layernorm25(x296)
        x298=self.linear50(x297)
        x299=self.gelu25(x298)
        x300=self.linear51(x299)
        return x300

m = M().eval()
x296 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x296)
end = time.time()
print(end-start)
