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
        self.layernorm11 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.linear22 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x142):
        x143=self.layernorm11(x142)
        x144=self.linear22(x143)
        return x144

m = M().eval()
x142 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x142)
end = time.time()
print(end-start)
