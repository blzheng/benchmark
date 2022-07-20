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
        self.layernorm13 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.linear26 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu13 = GELU(approximate='none')

    def forward(self, x164):
        x165=self.layernorm13(x164)
        x166=self.linear26(x165)
        x167=self.gelu13(x166)
        return x167

m = M().eval()
x164 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x164)
end = time.time()
print(end-start)
