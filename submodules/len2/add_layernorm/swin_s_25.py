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
        self.layernorm31 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)

    def forward(self, x333, x340):
        x341=operator.add(x333, x340)
        x342=self.layernorm31(x341)
        return x342

m = M().eval()
x333 = torch.randn(torch.Size([1, 14, 14, 384]))
x340 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x333, x340)
end = time.time()
print(end-start)
