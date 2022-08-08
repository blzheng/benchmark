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
        self.dropout32 = Dropout(p=0.1, inplace=False)
        self.layernorm21 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear64 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear65 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout33 = Dropout(p=0.1, inplace=False)
        self.layernorm22 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)

    def forward(self, x481, x448):
        x482=self.dropout32(x481)
        x483=operator.add(x482, x448)
        x484=self.layernorm21(x483)
        x485=self.linear64(x484)
        x486=torch._C._nn.gelu(x485)
        x487=self.linear65(x486)
        x488=self.dropout33(x487)
        x489=operator.add(x488, x484)
        x490=self.layernorm22(x489)
        return x490

m = M().eval()
x481 = torch.randn(torch.Size([1, 384, 768]))
x448 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x481, x448)
end = time.time()
print(end-start)
