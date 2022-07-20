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
        self.layernorm16 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)

    def forward(self, x203):
        x204=self.layernorm16(x203)
        return x204

m = M().eval()
x203 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x203)
end = time.time()
print(end-start)
