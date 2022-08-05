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
        self.layernorm32 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear30 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu14 = GELU(approximate='none')
        self.dropout28 = Dropout(p=0.0, inplace=False)
        self.linear31 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x356):
        x357=self.layernorm32(x356)
        x358=self.linear30(x357)
        x359=self.gelu14(x358)
        x360=self.dropout28(x359)
        x361=self.linear31(x360)
        return x361

m = M().eval()
x356 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x356)
end = time.time()
print(end-start)
