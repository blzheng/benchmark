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
        self.layernorm22 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear67 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x490):
        x491=self.layernorm22(x490)
        x492=self.linear67(x491)
        return x492

m = M().eval()
x490 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x490)
end = time.time()
print(end-start)
