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
        self.layernorm27 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear25 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x295):
        x296=self.layernorm27(x295)
        x297=self.linear25(x296)
        return x297

m = M().eval()
x295 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x295)
end = time.time()
print(end-start)
