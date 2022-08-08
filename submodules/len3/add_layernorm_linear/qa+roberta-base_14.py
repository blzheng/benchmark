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
        self.layernorm15 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear46 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x356, x322):
        x357=operator.add(x356, x322)
        x358=self.layernorm15(x357)
        x359=self.linear46(x358)
        return x359

m = M().eval()
x356 = torch.randn(torch.Size([1, 384, 768]))
x322 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x356, x322)
end = time.time()
print(end-start)