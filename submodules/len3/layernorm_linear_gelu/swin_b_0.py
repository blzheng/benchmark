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
        self.layernorm2 = LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        self.linear0 = Linear(in_features=128, out_features=512, bias=True)
        self.gelu0 = GELU(approximate='none')

    def forward(self, x18):
        x19=self.layernorm2(x18)
        x20=self.linear0(x19)
        x21=self.gelu0(x20)
        return x21

m = M().eval()
x18 = torch.randn(torch.Size([1, 56, 56, 128]))
start = time.time()
output = m(x18)
end = time.time()
print(end-start)
