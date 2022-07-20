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
        self.layernorm14 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.linear28 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x175):
        x176=self.layernorm14(x175)
        x177=self.linear28(x176)
        return x177

m = M().eval()
x175 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x175)
end = time.time()
print(end-start)
