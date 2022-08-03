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
        self.layernorm25 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)

    def forward(self, x257, x271):
        x272=operator.add(x257, x271)
        x273=self.layernorm25(x272)
        return x273

m = M().eval()
x257 = torch.randn(torch.Size([1, 7, 7, 768]))
x271 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x257, x271)
end = time.time()
print(end-start)
