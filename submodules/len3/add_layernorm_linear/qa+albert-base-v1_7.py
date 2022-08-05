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
        self.layernorm2 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear1 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x175, x172):
        x176=operator.add(x175, x172)
        x177=self.layernorm2(x176)
        x178=self.linear1(x177)
        return x178

m = M().eval()
x175 = torch.randn(torch.Size([1, 384, 768]))
x172 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x175, x172)
end = time.time()
print(end-start)
