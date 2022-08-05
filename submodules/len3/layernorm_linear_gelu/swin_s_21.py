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
        self.layernorm46 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear44 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu21 = GELU(approximate='none')

    def forward(self, x517):
        x518=self.layernorm46(x517)
        x519=self.linear44(x518)
        x520=self.gelu21(x519)
        return x520

m = M().eval()
x517 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x517)
end = time.time()
print(end-start)
