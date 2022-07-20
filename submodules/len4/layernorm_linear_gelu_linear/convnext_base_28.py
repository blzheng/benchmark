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
        self.layernorm28 = LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        self.linear56 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu28 = GELU(approximate='none')
        self.linear57 = Linear(in_features=2048, out_features=512, bias=True)

    def forward(self, x329):
        x330=self.layernorm28(x329)
        x331=self.linear56(x330)
        x332=self.gelu28(x331)
        x333=self.linear57(x332)
        return x333

m = M().eval()
x329 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x329)
end = time.time()
print(end-start)
