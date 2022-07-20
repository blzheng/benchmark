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
        self.layernorm2 = LayerNorm((192,), eps=1e-06, elementwise_affine=True)

    def forward(self, x31):
        x32=self.layernorm2(x31)
        return x32

m = M().eval()
x31 = torch.randn(torch.Size([1, 56, 56, 192]))
start = time.time()
output = m(x31)
end = time.time()
print(end-start)
