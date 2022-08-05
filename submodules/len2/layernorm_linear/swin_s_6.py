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
        self.layernorm12 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear10 = Linear(in_features=384, out_features=1536, bias=True)

    def forward(self, x126):
        x127=self.layernorm12(x126)
        x128=self.linear10(x127)
        return x128

m = M().eval()
x126 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x126)
end = time.time()
print(end-start)
