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
        self.layernorm26 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)

    def forward(self, x272, x279):
        x280=operator.add(x272, x279)
        x281=self.layernorm26(x280)
        return x281

m = M().eval()
x272 = torch.randn(torch.Size([1, 7, 7, 768]))
x279 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x272, x279)
end = time.time()
print(end-start)
