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
        self.layernorm29 = LayerNorm((384,), eps=1e-06, elementwise_affine=True)

    def forward(self, x340):
        x341=self.layernorm29(x340)
        return x341

m = M().eval()
x340 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x340)
end = time.time()
print(end-start)
