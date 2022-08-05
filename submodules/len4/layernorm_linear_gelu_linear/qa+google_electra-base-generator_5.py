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
        self.layernorm11 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear35 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear36 = Linear(in_features=1024, out_features=256, bias=True)

    def forward(self, x274):
        x275=self.layernorm11(x274)
        x276=self.linear35(x275)
        x277=torch._C._nn.gelu(x276)
        x278=self.linear36(x277)
        return x278

m = M().eval()
x274 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x274)
end = time.time()
print(end-start)
