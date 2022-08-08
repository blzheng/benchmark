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
        self.layernorm0 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)

    def forward(self, x20, x22, x25):
        x23=operator.add(x20, x22)
        x26=operator.add(x23, x25)
        x27=self.layernorm0(x26)
        return x27

m = M().eval()
x20 = torch.randn(torch.Size([1, 384, 768]))
x22 = torch.randn(torch.Size([1, 384, 768]))
x25 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x20, x22, x25)
end = time.time()
print(end-start)
