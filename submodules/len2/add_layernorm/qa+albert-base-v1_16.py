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
        self.layernorm2 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)

    def forward(self, x323, x320):
        x324=operator.add(x323, x320)
        x325=self.layernorm2(x324)
        return x325

m = M().eval()
x323 = torch.randn(torch.Size([1, 384, 768]))
x320 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x323, x320)
end = time.time()
print(end-start)
