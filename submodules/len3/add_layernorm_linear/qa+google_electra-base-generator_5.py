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
        self.layernorm6 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear19 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x153, x149):
        x154=operator.add(x153, x149)
        x155=self.layernorm6(x154)
        x156=self.linear19(x155)
        return x156

m = M().eval()
x153 = torch.randn(torch.Size([1, 384, 256]))
x149 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x153, x149)
end = time.time()
print(end-start)
