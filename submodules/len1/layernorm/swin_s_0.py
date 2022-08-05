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
        self.layernorm0 = LayerNorm((96,), eps=1e-05, elementwise_affine=True)

    def forward(self, x2):
        x3=self.layernorm0(x2)
        return x3

m = M().eval()
x2 = torch.randn(torch.Size([1, 56, 56, 96]))
start = time.time()
output = m(x2)
end = time.time()
print(end-start)
