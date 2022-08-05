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
        self.layernorm25 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear23 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu10 = GELU(approximate='none')
        self.dropout20 = Dropout(p=0.0, inplace=False)
        self.linear24 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x272):
        x273=self.layernorm25(x272)
        x274=self.linear23(x273)
        x275=self.gelu10(x274)
        x276=self.dropout20(x275)
        x277=self.linear24(x276)
        return x277

m = M().eval()
x272 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x272)
end = time.time()
print(end-start)
