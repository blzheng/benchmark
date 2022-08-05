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
        self.layernorm24 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear73 = Linear(in_features=256, out_features=2, bias=True)

    def forward(self, x532):
        x533=self.layernorm24(x532)
        x534=self.linear73(x533)
        return x534

m = M().eval()
x532 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x532)
end = time.time()
print(end-start)
