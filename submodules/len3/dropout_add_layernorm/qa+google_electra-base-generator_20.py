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
        self.layernorm21 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)

    def forward(self, x482, x449):
        x483=self.dropout32(x482)
        x484=operator.add(x483, x449)
        x485=self.layernorm21(x484)
        return x485

m = M().eval()
x482 = torch.randn(torch.Size([1, 384, 256]))
x449 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x482, x449)
end = time.time()
print(end-start)
