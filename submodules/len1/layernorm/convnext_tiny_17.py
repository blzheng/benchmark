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
        self.layernorm17 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)

    def forward(self, x214):
        x215=self.layernorm17(x214)
        return x215

m = M().eval()
x214 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x214)
end = time.time()
print(end-start)
