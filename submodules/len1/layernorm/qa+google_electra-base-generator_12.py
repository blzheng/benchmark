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
        self.layernorm12 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)

    def forward(self, x280):
        x281=self.layernorm12(x280)
        return x281

m = M().eval()
x280 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x280)
end = time.time()
print(end-start)
