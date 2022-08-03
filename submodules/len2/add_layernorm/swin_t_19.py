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
        self.layernorm27 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)

    def forward(self, x280, x294):
        x295=operator.add(x280, x294)
        x296=self.layernorm27(x295)
        return x296

m = M().eval()
x280 = torch.randn(torch.Size([1, 7, 7, 768]))
x294 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x280, x294)
end = time.time()
print(end-start)
