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
        self.layernorm2 = LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        self.linear0 = Linear(in_features=96, out_features=384, bias=True)
        self.gelu0 = GELU(approximate='none')
        self.dropout0 = Dropout(p=0.0, inplace=False)
        self.linear1 = Linear(in_features=384, out_features=96, bias=True)
        self.dropout1 = Dropout(p=0.0, inplace=False)

    def forward(self, x18):
        x19=self.layernorm2(x18)
        x20=self.linear0(x19)
        x21=self.gelu0(x20)
        x22=self.dropout0(x21)
        x23=self.linear1(x22)
        x24=self.dropout1(x23)
        return x24

m = M().eval()
x18 = torch.randn(torch.Size([1, 56, 56, 96]))
start = time.time()
output = m(x18)
end = time.time()
print(end-start)
