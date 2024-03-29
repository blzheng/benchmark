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
        self.layernorm8 = LayerNorm((192,), eps=1e-05, elementwise_affine=True)

    def forward(self, x80):
        x81=self.layernorm8(x80)
        return x81

m = M().eval()
x80 = torch.randn(torch.Size([1, 28, 28, 192]))
start = time.time()
output = m(x80)
end = time.time()
print(end-start)
