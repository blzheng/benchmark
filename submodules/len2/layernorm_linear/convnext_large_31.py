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
        self.layernorm31 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.linear62 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x362):
        x363=self.layernorm31(x362)
        x364=self.linear62(x363)
        return x364

m = M().eval()
x362 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x362)
end = time.time()
print(end-start)
