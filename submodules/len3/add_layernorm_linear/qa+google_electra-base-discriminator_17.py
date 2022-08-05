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
        self.layernorm18 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear54 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x404, x400):
        x405=operator.add(x404, x400)
        x406=self.layernorm18(x405)
        x407=self.linear54(x406)
        return x407

m = M().eval()
x404 = torch.randn(torch.Size([1, 384, 768]))
x400 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x404, x400)
end = time.time()
print(end-start)
