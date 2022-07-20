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
        self.layernorm35 = LayerNorm((1536,), eps=1e-06, elementwise_affine=True)

    def forward(self, x412):
        x413=self.layernorm35(x412)
        return x413

m = M().eval()
x412 = torch.randn(torch.Size([1, 7, 7, 1536]))
start = time.time()
output = m(x412)
end = time.time()
print(end-start)
