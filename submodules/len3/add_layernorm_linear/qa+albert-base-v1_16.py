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
        self.layernorm1 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear5 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x325, x355):
        x356=operator.add(x325, x355)
        x357=self.layernorm1(x356)
        x358=self.linear5(x357)
        return x358

m = M().eval()
x325 = torch.randn(torch.Size([1, 384, 768]))
x355 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x325, x355)
end = time.time()
print(end-start)
