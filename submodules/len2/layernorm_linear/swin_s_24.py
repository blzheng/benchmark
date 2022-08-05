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
        self.layernorm47 = LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
        self.linear46 = Linear(in_features=1536, out_features=768, bias=False)

    def forward(self, x531):
        x532=self.layernorm47(x531)
        x533=self.linear46(x532)
        return x533

m = M().eval()
x531 = torch.randn(torch.Size([1, 7, 7, 1536]))
start = time.time()
output = m(x531)
end = time.time()
print(end-start)
