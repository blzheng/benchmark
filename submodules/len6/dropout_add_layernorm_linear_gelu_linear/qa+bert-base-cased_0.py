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
        self.dropout0 = Dropout(p=0.1, inplace=False)
        self.layernorm1 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear4 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear5 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x27, x62):
        x28=self.dropout0(x27)
        x63=operator.add(x62, x28)
        x64=self.layernorm1(x63)
        x65=self.linear4(x64)
        x66=torch._C._nn.gelu(x65)
        x67=self.linear5(x66)
        return x67

m = M().eval()
x27 = torch.randn(torch.Size([1, 384, 768]))
x62 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x27, x62)
end = time.time()
print(end-start)
