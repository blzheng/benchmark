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
        self.layernorm20 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear18 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu8 = GELU(approximate='none')
        self.dropout16 = Dropout(p=0.0, inplace=False)
        self.linear19 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout17 = Dropout(p=0.0, inplace=False)

    def forward(self, x203, x217):
        x218=operator.add(x203, x217)
        x219=self.layernorm20(x218)
        x220=self.linear18(x219)
        x221=self.gelu8(x220)
        x222=self.dropout16(x221)
        x223=self.linear19(x222)
        x224=self.dropout17(x223)
        return x224

m = M().eval()
x203 = torch.randn(torch.Size([1, 14, 14, 384]))
x217 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x203, x217)
end = time.time()
print(end-start)
