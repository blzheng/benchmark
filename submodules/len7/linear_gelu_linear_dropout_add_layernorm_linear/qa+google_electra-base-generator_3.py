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
        self.linear23 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear24 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout12 = Dropout(p=0.1, inplace=False)
        self.layernorm8 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear25 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x191, x191):
        x192=self.linear23(x191)
        x193=torch._C._nn.gelu(x192)
        x194=self.linear24(x193)
        x195=self.dropout12(x194)
        x196=operator.add(x195, x191)
        x197=self.layernorm8(x196)
        x198=self.linear25(x197)
        return x198

m = M().eval()
x191 = torch.randn(torch.Size([1, 384, 256]))
x191 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x191, x191)
end = time.time()
print(end-start)
