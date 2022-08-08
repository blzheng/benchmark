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
        self.layernorm23 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)

    def forward(self, x524, x490):
        x525=operator.add(x524, x490)
        x526=self.layernorm23(x525)
        return x526

m = M().eval()
x524 = torch.randn(torch.Size([1, 384, 768]))
x490 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x524, x490)
end = time.time()
print(end-start)
