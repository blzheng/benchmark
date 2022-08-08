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
        self.layernorm5 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)

    def forward(self, x146, x112):
        x147=operator.add(x146, x112)
        x148=self.layernorm5(x147)
        return x148

m = M().eval()
x146 = torch.randn(torch.Size([1, 384, 768]))
x112 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x146, x112)
end = time.time()
print(end-start)
