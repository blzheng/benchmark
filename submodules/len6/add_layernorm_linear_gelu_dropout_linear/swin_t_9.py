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
        self.layernorm22 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear20 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu9 = GELU(approximate='none')
        self.dropout18 = Dropout(p=0.0, inplace=False)
        self.linear21 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x226, x240):
        x241=operator.add(x226, x240)
        x242=self.layernorm22(x241)
        x243=self.linear20(x242)
        x244=self.gelu9(x243)
        x245=self.dropout18(x244)
        x246=self.linear21(x245)
        return x246

m = M().eval()
x226 = torch.randn(torch.Size([1, 14, 14, 384]))
x240 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x226, x240)
end = time.time()
print(end-start)
