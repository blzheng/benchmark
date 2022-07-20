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
        self.layernorm9 = LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        self.linear18 = Linear(in_features=384, out_features=1536, bias=True)

    def forward(self, x120):
        x121=self.layernorm9(x120)
        x122=self.linear18(x121)
        return x122

m = M().eval()
x120 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x120)
end = time.time()
print(end-start)
