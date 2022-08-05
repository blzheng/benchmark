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

    def forward(self, x139):
        x140=self.layernorm2(x139)
        x141=self.linear1(x140)
        return x141

m = M().eval()
x139 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x139)
end = time.time()
print(end-start)
