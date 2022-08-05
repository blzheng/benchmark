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

    def forward(self, x241):
        x242=self.layernorm22(x241)
        x243=self.linear20(x242)
        x244=self.gelu9(x243)
        return x244

m = M().eval()
x241 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x241)
end = time.time()
print(end-start)
