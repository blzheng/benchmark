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
        self.layernorm37 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)

    def forward(self, x402, x409):
        x410=operator.add(x402, x409)
        x411=self.layernorm37(x410)
        return x411

m = M().eval()
x402 = torch.randn(torch.Size([1, 14, 14, 384]))
x409 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x402, x409)
end = time.time()
print(end-start)
