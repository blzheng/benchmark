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
        self.layernorm18 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear16 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu7 = GELU(approximate='none')
        self.dropout14 = Dropout(p=0.0, inplace=False)
        self.linear17 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout15 = Dropout(p=0.0, inplace=False)

    def forward(self, x180, x194):
        x195=operator.add(x180, x194)
        x196=self.layernorm18(x195)
        x197=self.linear16(x196)
        x198=self.gelu7(x197)
        x199=self.dropout14(x198)
        x200=self.linear17(x199)
        x201=self.dropout15(x200)
        return x201

m = M().eval()
x180 = torch.randn(torch.Size([1, 14, 14, 512]))
x194 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x180, x194)
end = time.time()
print(end-start)
