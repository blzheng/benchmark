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
        self.layernorm4 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)

    def forward(self, x110, x106):
        x111=operator.add(x110, x106)
        x112=self.layernorm4(x111)
        return x112

m = M().eval()
x110 = torch.randn(torch.Size([1, 384, 768]))
x106 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x110, x106)
end = time.time()
print(end-start)
