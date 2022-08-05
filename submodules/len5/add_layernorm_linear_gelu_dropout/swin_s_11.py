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
        self.layernorm26 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear24 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu11 = GELU(approximate='none')
        self.dropout22 = Dropout(p=0.0, inplace=False)

    def forward(self, x272, x286):
        x287=operator.add(x272, x286)
        x288=self.layernorm26(x287)
        x289=self.linear24(x288)
        x290=self.gelu11(x289)
        x291=self.dropout22(x290)
        return x291

m = M().eval()
x272 = torch.randn(torch.Size([1, 14, 14, 384]))
x286 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x272, x286)
end = time.time()
print(end-start)
