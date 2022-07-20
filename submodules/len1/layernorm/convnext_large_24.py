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
        self.layernorm24 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)

    def forward(self, x285):
        x286=self.layernorm24(x285)
        return x286

m = M().eval()
x285 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x285)
end = time.time()
print(end-start)
