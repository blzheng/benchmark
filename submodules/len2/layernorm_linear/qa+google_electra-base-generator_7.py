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
        self.layernorm8 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear25 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x196):
        x197=self.layernorm8(x196)
        x198=self.linear25(x197)
        return x198

m = M().eval()
x196 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x196)
end = time.time()
print(end-start)
