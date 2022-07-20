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
        self.layernorm1 = LayerNorm((128,), eps=1e-06, elementwise_affine=True)
        self.linear2 = Linear(in_features=128, out_features=512, bias=True)
        self.gelu1 = GELU(approximate='none')
        self.linear3 = Linear(in_features=512, out_features=128, bias=True)

    def forward(self, x20):
        x21=self.layernorm1(x20)
        x22=self.linear2(x21)
        x23=self.gelu1(x22)
        x24=self.linear3(x23)
        return x24

m = M().eval()
x20 = torch.randn(torch.Size([1, 56, 56, 128]))
start = time.time()
output = m(x20)
end = time.time()
print(end-start)
