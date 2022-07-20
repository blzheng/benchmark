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
        self.layernorm33 = LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
        self.linear66 = Linear(in_features=1536, out_features=6144, bias=True)
        self.gelu33 = GELU(approximate='none')

    def forward(self, x390):
        x391=self.layernorm33(x390)
        x392=self.linear66(x391)
        x393=self.gelu33(x392)
        return x393

m = M().eval()
x390 = torch.randn(torch.Size([1, 7, 7, 1536]))
start = time.time()
output = m(x390)
end = time.time()
print(end-start)
