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
        self.layernorm15 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)

    def forward(self, x356, x322):
        x357=operator.add(x356, x322)
        x358=self.layernorm15(x357)
        return x358

m = M().eval()
x356 = torch.randn(torch.Size([1, 384, 768]))
x322 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x356, x322)
end = time.time()
print(end-start)
