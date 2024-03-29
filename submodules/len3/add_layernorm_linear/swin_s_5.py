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
        self.layernorm14 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear12 = Linear(in_features=384, out_features=1536, bias=True)

    def forward(self, x134, x148):
        x149=operator.add(x134, x148)
        x150=self.layernorm14(x149)
        x151=self.linear12(x150)
        return x151

m = M().eval()
x134 = torch.randn(torch.Size([1, 14, 14, 384]))
x148 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x134, x148)
end = time.time()
print(end-start)
