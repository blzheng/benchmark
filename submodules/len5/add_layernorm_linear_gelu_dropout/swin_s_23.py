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
        self.gelu23 = GELU(approximate='none')
        self.dropout46 = Dropout(p=0.0, inplace=False)

    def forward(self, x556, x570):
        x571=operator.add(x556, x570)
        x572=self.layernorm51(x571)
        x573=self.linear49(x572)
        x574=self.gelu23(x573)
        x575=self.dropout46(x574)
        return x575

m = M().eval()
x556 = torch.randn(torch.Size([1, 7, 7, 768]))
x570 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x556, x570)
end = time.time()
print(end-start)
