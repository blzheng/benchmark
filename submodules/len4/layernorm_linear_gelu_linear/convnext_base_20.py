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
        self.layernorm20 = LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        self.linear40 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu20 = GELU(approximate='none')
        self.linear41 = Linear(in_features=2048, out_features=512, bias=True)

    def forward(self, x241):
        x242=self.layernorm20(x241)
        x243=self.linear40(x242)
        x244=self.gelu20(x243)
        x245=self.linear41(x244)
        return x245

m = M().eval()
x241 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x241)
end = time.time()
print(end-start)
