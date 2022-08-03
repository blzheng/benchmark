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

    def forward(self, x18):
        x19=self.layernorm2(x18)
        x20=self.linear0(x19)
        return x20

m = M().eval()
x18 = torch.randn(torch.Size([1, 56, 56, 96]))
start = time.time()
output = m(x18)
end = time.time()
print(end-start)
