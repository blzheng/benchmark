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
        self.layernorm30 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear28 = Linear(in_features=384, out_features=1536, bias=True)

    def forward(self, x318, x332):
        x333=operator.add(x318, x332)
        x334=self.layernorm30(x333)
        x335=self.linear28(x334)
        return x335

m = M().eval()
x318 = torch.randn(torch.Size([1, 14, 14, 384]))
x332 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x318, x332)
end = time.time()
print(end-start)
