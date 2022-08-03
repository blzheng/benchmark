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
        self.layernorm23 = LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
        self.linear22 = Linear(in_features=1536, out_features=768, bias=False)

    def forward(self, x255):
        x256=self.layernorm23(x255)
        x257=self.linear22(x256)
        return x257

m = M().eval()
x255 = torch.randn(torch.Size([1, 7, 7, 1536]))
start = time.time()
output = m(x255)
end = time.time()
print(end-start)
