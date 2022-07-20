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
        self.layernorm5 = LayerNorm((256,), eps=1e-06, elementwise_affine=True)
        self.linear10 = Linear(in_features=256, out_features=1024, bias=True)
        self.gelu5 = GELU(approximate='none')
        self.linear11 = Linear(in_features=1024, out_features=256, bias=True)

    def forward(self, x70):
        x71=self.layernorm5(x70)
        x72=self.linear10(x71)
        x73=self.gelu5(x72)
        x74=self.linear11(x73)
        return x74

m = M().eval()
x70 = torch.randn(torch.Size([1, 28, 28, 256]))
start = time.time()
output = m(x70)
end = time.time()
print(end-start)
