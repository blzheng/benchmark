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
        self.layernorm10 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)

    def forward(self, x237, x233):
        x238=operator.add(x237, x233)
        x239=self.layernorm10(x238)
        return x239

m = M().eval()
x237 = torch.randn(torch.Size([1, 384, 256]))
x233 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x237, x233)
end = time.time()
print(end-start)
