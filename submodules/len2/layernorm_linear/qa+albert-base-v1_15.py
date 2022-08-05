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
        self.layernorm2 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear1 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x324):
        x325=self.layernorm2(x324)
        x326=self.linear1(x325)
        return x326

m = M().eval()
x324 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x324)
end = time.time()
print(end-start)
