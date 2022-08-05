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
        self.layernorm52 = LayerNorm((1024,), eps=1e-05, elementwise_affine=True)

    def forward(self, x571, x578):
        x579=operator.add(x571, x578)
        x580=self.layernorm52(x579)
        return x580

m = M().eval()
x571 = torch.randn(torch.Size([1, 7, 7, 1024]))
x578 = torch.randn(torch.Size([1, 7, 7, 1024]))
start = time.time()
output = m(x571, x578)
end = time.time()
print(end-start)
