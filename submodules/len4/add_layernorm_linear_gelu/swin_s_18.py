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
        self.layernorm40 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear38 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu18 = GELU(approximate='none')

    def forward(self, x433, x447):
        x448=operator.add(x433, x447)
        x449=self.layernorm40(x448)
        x450=self.linear38(x449)
        x451=self.gelu18(x450)
        return x451

m = M().eval()
x433 = torch.randn(torch.Size([1, 14, 14, 384]))
x447 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x433, x447)
end = time.time()
print(end-start)
