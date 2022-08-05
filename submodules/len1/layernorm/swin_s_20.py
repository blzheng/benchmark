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
        self.layernorm20 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)

    def forward(self, x218):
        x219=self.layernorm20(x218)
        return x219

m = M().eval()
x218 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x218)
end = time.time()
print(end-start)
