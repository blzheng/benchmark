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
        self.layernorm21 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear64 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear65 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout33 = Dropout(p=0.1, inplace=False)

    def forward(self, x482, x448):
        x483=operator.add(x482, x448)
        x484=self.layernorm21(x483)
        x485=self.linear64(x484)
        x486=torch._C._nn.gelu(x485)
        x487=self.linear65(x486)
        x488=self.dropout33(x487)
        x489=operator.add(x488, x484)
        return x489

m = M().eval()
x482 = torch.randn(torch.Size([1, 384, 768]))
x448 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x482, x448)
end = time.time()
print(end-start)
