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
        self.layernorm19 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear58 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x441):
        x442=self.layernorm19(x441)
        x443=self.linear58(x442)
        return x443

m = M().eval()
x441 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x441)
end = time.time()
print(end-start)
