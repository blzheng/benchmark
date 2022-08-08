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
        self.layernorm32 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear30 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu14 = GELU(approximate='none')

    def forward(self, x341, x355):
        x356=operator.add(x341, x355)
        x357=self.layernorm32(x356)
        x358=self.linear30(x357)
        x359=self.gelu14(x358)
        return x359

m = M().eval()
x341 = torch.randn(torch.Size([1, 14, 14, 512]))
x355 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x341, x355)
end = time.time()
print(end-start)