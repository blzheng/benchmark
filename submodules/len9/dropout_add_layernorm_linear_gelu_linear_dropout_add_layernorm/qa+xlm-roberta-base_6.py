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
        self.dropout17 = Dropout(p=0.1, inplace=False)
        self.layernorm11 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear34 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear35 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout18 = Dropout(p=0.1, inplace=False)
        self.layernorm12 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)

    def forward(self, x271, x238):
        x272=self.dropout17(x271)
        x273=operator.add(x272, x238)
        x274=self.layernorm11(x273)
        x275=self.linear34(x274)
        x276=torch._C._nn.gelu(x275)
        x277=self.linear35(x276)
        x278=self.dropout18(x277)
        x279=operator.add(x278, x274)
        x280=self.layernorm12(x279)
        return x280

m = M().eval()
x271 = torch.randn(torch.Size([1, 384, 768]))
x238 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x271, x238)
end = time.time()
print(end-start)
