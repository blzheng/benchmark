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
        self.layernorm3 = LayerNorm((96,), eps=1e-05, elementwise_affine=True)

    def forward(self, x18, x25):
        x26=operator.add(x18, x25)
        x27=self.layernorm3(x26)
        return x27

m = M().eval()
x18 = torch.randn(torch.Size([1, 56, 56, 96]))
x25 = torch.randn(torch.Size([1, 56, 56, 96]))
start = time.time()
output = m(x18, x25)
end = time.time()
print(end-start)
