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
        self.layernorm16 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)

    def forward(self, x157, x171):
        x172=operator.add(x157, x171)
        x173=self.layernorm16(x172)
        return x173

m = M().eval()
x157 = torch.randn(torch.Size([1, 14, 14, 384]))
x171 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x157, x171)
end = time.time()
print(end-start)
