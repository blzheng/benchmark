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
        self.linear46 = Linear(in_features=1536, out_features=768, bias=False)
        self.layernorm48 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)

    def forward(self, x532):
        x533=self.linear46(x532)
        x534=self.layernorm48(x533)
        return x534

m = M().eval()
x532 = torch.randn(torch.Size([1, 7, 7, 1536]))
start = time.time()
output = m(x532)
end = time.time()
print(end-start)
