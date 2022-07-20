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
        self.layernorm16 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.linear32 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu16 = GELU(approximate='none')

    def forward(self, x203):
        x204=self.layernorm16(x203)
        x205=self.linear32(x204)
        x206=self.gelu16(x205)
        return x206

m = M().eval()
x203 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x203)
end = time.time()
print(end-start)
