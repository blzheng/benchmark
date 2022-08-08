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
        self.linear4 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout2 = Dropout(p=0.1, inplace=False)
        self.layernorm1 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear5 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear6 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x242, x214):
        x243=self.linear4(x242)
        x244=self.dropout2(x243)
        x245=operator.add(x214, x244)
        x246=self.layernorm1(x245)
        x247=self.linear5(x246)
        x248=torch._C._nn.gelu(x247)
        x249=self.linear6(x248)
        return x249

m = M().eval()
x242 = torch.randn(torch.Size([1, 384, 768]))
x214 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x242, x214)
end = time.time()
print(end-start)
