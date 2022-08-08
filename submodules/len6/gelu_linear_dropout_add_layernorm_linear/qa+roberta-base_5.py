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
        self.linear35 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout18 = Dropout(p=0.1, inplace=False)
        self.layernorm12 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear36 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x275, x274):
        x276=torch._C._nn.gelu(x275)
        x277=self.linear35(x276)
        x278=self.dropout18(x277)
        x279=operator.add(x278, x274)
        x280=self.layernorm12(x279)
        x281=self.linear36(x280)
        return x281

m = M().eval()
x275 = torch.randn(torch.Size([1, 384, 3072]))
x274 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x275, x274)
end = time.time()
print(end-start)
