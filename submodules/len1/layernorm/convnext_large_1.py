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
        self.layernorm1 = LayerNorm((192,), eps=1e-06, elementwise_affine=True)

    def forward(self, x20):
        x21=self.layernorm1(x20)
        return x21

m = M().eval()
x20 = torch.randn(torch.Size([1, 56, 56, 192]))
start = time.time()
output = m(x20)
end = time.time()
print(end-start)
