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
        self.layernorm51 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear49 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x571):
        x572=self.layernorm51(x571)
        x573=self.linear49(x572)
        return x573

m = M().eval()
x571 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x571)
end = time.time()
print(end-start)
