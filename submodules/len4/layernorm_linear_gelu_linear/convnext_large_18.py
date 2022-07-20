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
        self.layernorm18 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.linear36 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu18 = GELU(approximate='none')
        self.linear37 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x219):
        x220=self.layernorm18(x219)
        x221=self.linear36(x220)
        x222=self.gelu18(x221)
        x223=self.linear37(x222)
        return x223

m = M().eval()
x219 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x219)
end = time.time()
print(end-start)
