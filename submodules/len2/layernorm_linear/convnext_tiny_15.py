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
        self.layernorm15 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.linear30 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x192):
        x193=self.layernorm15(x192)
        x194=self.linear30(x193)
        return x194

m = M().eval()
x192 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x192)
end = time.time()
print(end-start)
