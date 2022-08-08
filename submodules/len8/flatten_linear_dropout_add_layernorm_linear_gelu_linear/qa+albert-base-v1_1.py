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

    def forward(self, x93, x66):
        x94=x93.flatten(2)
        x95=self.linear4(x94)
        x96=self.dropout2(x95)
        x97=operator.add(x66, x96)
        x98=self.layernorm1(x97)
        x99=self.linear5(x98)
        x100=torch._C._nn.gelu(x99)
        x101=self.linear6(x100)
        return x101

m = M().eval()
x93 = torch.randn(torch.Size([1, 384, 12, 64]))
x66 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x93, x66)
end = time.time()
print(end-start)
