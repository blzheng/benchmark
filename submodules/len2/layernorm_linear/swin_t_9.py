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
        self.layernorm18 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear16 = Linear(in_features=384, out_features=1536, bias=True)

    def forward(self, x195):
        x196=self.layernorm18(x195)
        x197=self.linear16(x196)
        return x197

m = M().eval()
x195 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x195)
end = time.time()
print(end-start)
