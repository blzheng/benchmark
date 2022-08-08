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
        self.dropout2 = Dropout(p=0.1, inplace=False)
        self.layernorm1 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear5 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear6 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout3 = Dropout(p=0.1, inplace=False)
        self.layernorm2 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear7 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x62, x29):
        x63=self.dropout2(x62)
        x64=operator.add(x63, x29)
        x65=self.layernorm1(x64)
        x66=self.linear5(x65)
        x67=torch._C._nn.gelu(x66)
        x68=self.linear6(x67)
        x69=self.dropout3(x68)
        x70=operator.add(x69, x65)
        x71=self.layernorm2(x70)
        x72=self.linear7(x71)
        return x72

m = M().eval()
x62 = torch.randn(torch.Size([1, 384, 256]))
x29 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x62, x29)
end = time.time()
print(end-start)
