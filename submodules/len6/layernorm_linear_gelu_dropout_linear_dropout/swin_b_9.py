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
        self.layernorm22 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear20 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu9 = GELU(approximate='none')
        self.dropout18 = Dropout(p=0.0, inplace=False)
        self.linear21 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout19 = Dropout(p=0.0, inplace=False)

    def forward(self, x241):
        x242=self.layernorm22(x241)
        x243=self.linear20(x242)
        x244=self.gelu9(x243)
        x245=self.dropout18(x244)
        x246=self.linear21(x245)
        x247=self.dropout19(x246)
        return x247

m = M().eval()
x241 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x241)
end = time.time()
print(end-start)
