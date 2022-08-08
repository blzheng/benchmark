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
        self.dropout11 = Dropout(p=0.1, inplace=False)
        self.layernorm7 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear22 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear23 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout12 = Dropout(p=0.1, inplace=False)
        self.layernorm8 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear24 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x187, x154):
        x188=self.dropout11(x187)
        x189=operator.add(x188, x154)
        x190=self.layernorm7(x189)
        x191=self.linear22(x190)
        x192=torch._C._nn.gelu(x191)
        x193=self.linear23(x192)
        x194=self.dropout12(x193)
        x195=operator.add(x194, x190)
        x196=self.layernorm8(x195)
        x197=self.linear24(x196)
        return x197

m = M().eval()
x187 = torch.randn(torch.Size([1, 384, 768]))
x154 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x187, x154)
end = time.time()
print(end-start)
