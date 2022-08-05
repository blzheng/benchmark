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
        self.layernorm1 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear5 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x206, x177):
        x207=self.dropout2(x206)
        x208=operator.add(x177, x207)
        x209=self.layernorm1(x208)
        x210=self.linear5(x209)
        return x210

m = M().eval()
x206 = torch.randn(torch.Size([1, 384, 768]))
x177 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x206, x177)
end = time.time()
print(end-start)
