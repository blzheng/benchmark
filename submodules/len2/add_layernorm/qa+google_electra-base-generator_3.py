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
        self.layernorm3 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)

    def forward(self, x105, x71):
        x106=operator.add(x105, x71)
        x107=self.layernorm3(x106)
        return x107

m = M().eval()
x105 = torch.randn(torch.Size([1, 384, 256]))
x71 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x105, x71)
end = time.time()
print(end-start)
