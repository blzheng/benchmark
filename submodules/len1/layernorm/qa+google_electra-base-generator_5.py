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
        self.layernorm5 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)

    def forward(self, x148):
        x149=self.layernorm5(x148)
        return x149

m = M().eval()
x148 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x148)
end = time.time()
print(end-start)
