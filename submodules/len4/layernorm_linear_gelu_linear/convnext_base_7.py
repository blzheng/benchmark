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
        self.layernorm7 = LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        self.linear14 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu7 = GELU(approximate='none')
        self.linear15 = Linear(in_features=2048, out_features=512, bias=True)

    def forward(self, x98):
        x99=self.layernorm7(x98)
        x100=self.linear14(x99)
        x101=self.gelu7(x100)
        x102=self.linear15(x101)
        return x102

m = M().eval()
x98 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x98)
end = time.time()
print(end-start)
