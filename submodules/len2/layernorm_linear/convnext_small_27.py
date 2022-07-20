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
        self.layernorm27 = LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        self.linear54 = Linear(in_features=384, out_features=1536, bias=True)

    def forward(self, x318):
        x319=self.layernorm27(x318)
        x320=self.linear54(x319)
        return x320

m = M().eval()
x318 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x318)
end = time.time()
print(end-start)
