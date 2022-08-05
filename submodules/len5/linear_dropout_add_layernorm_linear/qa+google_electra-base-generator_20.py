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
        self.linear64 = Linear(in_features=256, out_features=256, bias=True)
        self.dropout32 = Dropout(p=0.1, inplace=False)
        self.layernorm21 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear65 = Linear(in_features=256, out_features=1024, bias=True)

    def forward(self, x481, x449):
        x482=self.linear64(x481)
        x483=self.dropout32(x482)
        x484=operator.add(x483, x449)
        x485=self.layernorm21(x484)
        x486=self.linear65(x485)
        return x486

m = M().eval()
x481 = torch.randn(torch.Size([1, 384, 256]))
x449 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x481, x449)
end = time.time()
print(end-start)
