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
        self.gelu13 = GELU(approximate='none')

    def forward(self, x333):
        x334=self.layernorm30(x333)
        x335=self.linear28(x334)
        x336=self.gelu13(x335)
        return x336

m = M().eval()
x333 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x333)
end = time.time()
print(end-start)
